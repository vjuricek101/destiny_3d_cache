import torch
import math

def calculate_divisibility_penalty(num_log2, den_log2):
    """
    Differentiable penalty enforcing that 2^num_log2 is perfectly divisible by 2^den_log2.
    Requires num_log2 - den_log2 >= 0, and num_log2 - den_log2 is an integer.
    """
    diff = num_log2 - den_log2
    neg_penalty = torch.relu(-diff)
    frac = diff - torch.round(diff.detach())
    int_penalty = frac ** 2
    return neg_penalty + int_penalty

def calculate_rows_per_subarray_penalty(cap_log10, ww_log2, assoc_log2, mat_c_log2, sub_c_log2):
    """
    Differentiable physics penalty enforcing the data-array rows-per-subarray identity.

    DESTINY requires (BankWithoutHtree.cpp:113, BankWithHtree.cpp:381):
        rows_per_subarray = (cap_bytes * 8) / (assoc * wordWidth * numActiveMatPerCol * numActiveSubarrayPerCol)

    In log2-space this is:
        log2_rps = log2(cap_bits) - log2(assoc) - log2(ww) - log2(nmac_col) - log2(nsac_col)

    Two soft sub-penalties are summed (caller applies a scalar weight):
        relu(-log2_rps)                               -- negativity: not enough rows
        (log2_rps - round(log2_rps).detach())^2      -- divisibility: non-integer log2
    """
    LOG2_10 = 3.321928094887362
    log2_cap_bits = cap_log10 * LOG2_10 + 13.0
    log2_rps = log2_cap_bits - assoc_log2 - ww_log2 - mat_c_log2 - sub_c_log2

    neg_penalty = torch.relu(-log2_rps)
    frac = log2_rps - torch.round(log2_rps.detach())
    int_penalty = frac ** 2

    return neg_penalty + int_penalty


def calculate_partition_penalty(ww_phys, mat_r_phys, mat_c_phys, sub_r_phys, sub_c_phys):
    """
    DESTINY partition constraint penalty.
    Enforces: word_width >= mat_r * mat_c * sub_r * sub_c
    From DESTINY main.cpp: blockSize / (active_partition_product) == 0
    causes every candidate config to be silently skipped (numSolutions = 0).
    """
    partition_phys = mat_r_phys * mat_c_phys * sub_r_phys * sub_c_phys
    return torch.relu(partition_phys - ww_phys) / ww_phys


def compute_physics_penalties(encoded_vals_dict, fixed_context, cap_enc, device):
    """
    Computes and aggregates all physics penalties (partition limits and row/subarray geometry constraints).
    Expects encoded_vals_dict to map parameter names to their differentiable encoded scalar tensors.
    Subarray geometry entries are expected to be linear scalars.
    """
    _ww_enc = encoded_vals_dict.get(
        "word_width_bits",
        torch.tensor(math.log2(float(fixed_context.get("word_width_bits", 64))),
                    device=device)
    )
    _mat_r_enc = encoded_vals_dict.get(
        "data_num_active_mat_per_row",
        torch.tensor(0.0, device=device)   # log2(1) = 0 → physical default = 1
    )
    _mat_c_enc = encoded_vals_dict.get(
        "data_num_active_mat_per_col",
        torch.tensor(0.0, device=device)   # log2(1) = 0 → physical default = 1
    )
    
    # Subarray terms: linear vocab {1.0, 2.0}, already physical, clamped to min 1.0 for stability
    _sub_r_phys = encoded_vals_dict.get(
        "data_num_active_subarray_per_row",
        torch.tensor(1.0, device=device)
    ).clamp(min=1.0)
    
    _sub_c_phys = encoded_vals_dict.get(
        "data_num_active_subarray_per_col",
        torch.tensor(1.0, device=device)
    ).clamp(min=1.0)

    _ww_phys        = 2.0 ** _ww_enc
    _mat_r_phys     = 2.0 ** _mat_r_enc
    _mat_c_phys     = 2.0 ** _mat_c_enc
    
    partition_penalty = 50.0 * calculate_partition_penalty(_ww_phys, _mat_r_phys, _mat_c_phys, _sub_r_phys, _sub_c_phys)

    # Active <= Total penalty for Data Mats
    _mat_total_r_enc = encoded_vals_dict.get("data_num_row_mat", _mat_r_enc)
    _mat_total_c_enc = encoded_vals_dict.get("data_num_col_mat", _mat_c_enc)
    active_vs_total_penalty = torch.relu(_mat_r_enc - _mat_total_r_enc) + torch.relu(_mat_c_enc - _mat_total_c_enc)

    # Tag arrays penalty
    _tag_mat_r_enc = encoded_vals_dict.get("tag_num_active_mat_per_row", torch.tensor(0.0, device=device))
    _tag_mat_c_enc = encoded_vals_dict.get("tag_num_active_mat_per_col", torch.tensor(0.0, device=device))
    _tag_mat_total_r_enc = encoded_vals_dict.get("tag_num_col_mat", _tag_mat_r_enc)
    _tag_mat_total_c_enc = encoded_vals_dict.get("tag_num_row_mat", _tag_mat_c_enc)
    
    tag_active_vs_total_penalty = torch.relu(_tag_mat_r_enc - _tag_mat_total_r_enc) + torch.relu(_tag_mat_c_enc - _tag_mat_total_c_enc)
    
    _tag_sub_r_phys = encoded_vals_dict.get("tag_num_row_subarray", torch.tensor(1.0, device=device)).clamp(min=1.0)
    _tag_sub_c_phys = encoded_vals_dict.get("tag_num_col_subarray", torch.tensor(1.0, device=device)).clamp(min=1.0)
    _tag_sub_r_active_phys = encoded_vals_dict.get("tag_num_active_subarray_per_row", _tag_sub_r_phys).clamp(min=1.0)
    _tag_sub_c_active_phys = encoded_vals_dict.get("tag_num_active_subarray_per_col", _tag_sub_c_phys).clamp(min=1.0)
    _tag_sub_r_active_log2 = torch.log2(_tag_sub_r_active_phys)
    _tag_sub_c_active_log2 = torch.log2(_tag_sub_c_active_phys)
    
    _tag_mat_r_phys = 2.0 ** _tag_mat_r_enc
    _tag_mat_c_phys = 2.0 ** _tag_mat_c_enc
    
    # Tag approx word width log2 (DESTINY usually has ~30-34 bit tag width for typical caches)
    # 32 bits = log2(32) = 5.0
    _tag_ww_enc = torch.tensor(5.0, device=device)
    
    # Approximate tag word width for partition penalty
    tag_partition_penalty = 50.0 * calculate_partition_penalty(2.0 ** _tag_ww_enc, _tag_mat_r_phys, _tag_mat_c_phys, _tag_sub_r_phys, _tag_sub_c_phys)

    _assoc_enc = encoded_vals_dict.get(
        "associativity", 
        torch.tensor(math.log2(float(fixed_context.get("associativity", 4))), device=device)
    )
    _sub_r_log2 = torch.log2(_sub_r_phys)
    _sub_c_log2 = torch.log2(_sub_c_phys)
    _tag_sub_r_log2 = torch.log2(_tag_sub_r_phys)
    _tag_sub_c_log2 = torch.log2(_tag_sub_c_phys)
    
    rows_penalty = 20.0 * calculate_rows_per_subarray_penalty(cap_enc, _ww_enc, _assoc_enc, _mat_c_enc, _sub_c_log2)
    
    # Strict Divisibility Constraints (To prevent DESTINY truncation failures)
    LOG2_10 = 3.321928094887362
    cap_bits_log2 = cap_enc * LOG2_10 + 13.0
    
    # 1. Capacity Divisibility (Total Mats)
    # matBlockSize = capacity_bits / (Total Row Mats * Total Col Mats)
    data_total_mats_log2 = _mat_total_r_enc + _mat_total_c_enc
    data_cap_div_penalty = 50.0 * calculate_divisibility_penalty(cap_bits_log2, data_total_mats_log2)
    
    tag_total_mats_log2 = _tag_mat_total_r_enc + _tag_mat_total_c_enc
    tag_cap_div_penalty = 50.0 * calculate_divisibility_penalty(cap_bits_log2, tag_total_mats_log2)
    
    # 2. Word Width Divisibility (Active Partitions)
    # numColumn truncates if wordWidth / (active partitions) is not an integer
    data_active_partitions_log2 = _mat_r_enc + _mat_c_enc + _sub_r_log2 + _sub_c_log2
    data_ww_div_penalty = 50.0 * calculate_divisibility_penalty(_ww_enc, data_active_partitions_log2)
    tag_active_partitions_log2 = _tag_mat_r_enc + _tag_mat_c_enc + _tag_sub_r_active_log2 + _tag_sub_c_active_log2
    tag_ww_div_penalty = 50.0 * calculate_divisibility_penalty(_tag_ww_enc, tag_active_partitions_log2)
    
    # 3. Tag Associativity Divisibility (Active Mats)
    # Tag array distributes associativity (Ways) across active mats. Truncation to 0 fails the design.
    tag_active_mats_log2 = _tag_mat_r_enc + _tag_mat_c_enc
    tag_assoc_div_penalty = 100.0 * calculate_divisibility_penalty(_assoc_enc, tag_active_mats_log2)
    
    # 4. Address Bits Remaining Constraint
    # DESTINY routes inactive partitions by consuming address bits. If all address bits are consumed, it fails with "numAddressBit <= 0".
    # Data Array: Address bits = log2(Capacity_Bits / WordWidth)
    data_initial_address_bits = cap_bits_log2 - _ww_enc
    data_address_bits_consumed = (_mat_total_r_enc + _mat_total_c_enc) - (_mat_r_enc + _mat_c_enc) + (_sub_r_log2 + _sub_c_log2) - (_sub_r_log2 + _sub_c_log2)
    # Wait, for Data we don't have separate sub total/active in this script, so sub bits consumed is 0.
    data_address_bits_remaining = data_initial_address_bits - data_address_bits_consumed
    # Must be > 0. (>= 1 in discrete space)
    data_addr_penalty = 50.0 * torch.relu(1.0 - data_address_bits_remaining)
    
    # Tag Array: Address bits = log2(Capacity_Bits / WordWidth / Assoc)
    # In DESTINY, Tag initial capacity passed to Bank is (numDataSet * Assoc * TagWidth).
    # Then BankWithHtree divides by TagWidth and Assoc.
    # So initial address bits = log2(numDataSet * Assoc / Assoc) = log2(numDataSet).
    # numDataSet = Capacity_Bytes / (LineSize * Assoc) = Capacity_Bits / (WordWidth * Assoc)
    tag_initial_address_bits = cap_bits_log2 - _ww_enc - _assoc_enc
    # Tag inactive mats + Tag inactive subarrays
    tag_address_bits_consumed = (_tag_mat_total_r_enc + _tag_mat_total_c_enc) - (_tag_mat_r_enc + _tag_mat_c_enc) + (_tag_sub_r_log2 + _tag_sub_c_log2) - (_tag_sub_r_active_log2 + _tag_sub_c_active_log2)
    tag_address_bits_remaining = tag_initial_address_bits - tag_address_bits_consumed
    tag_addr_penalty = 100.0 * torch.relu(1.0 - tag_address_bits_remaining)
    
    # 5. Tag Active vs Total Subarrays (Must be equal for capacity matching)
    # The Tag array does not divide numRow by active subarrays in Mat.cpp.
    # So capacity_calculated diverges if total_subarrays != active_subarrays.
    tag_sub_active_vs_total_penalty = 100.0 * torch.relu(_tag_sub_r_log2 - _tag_sub_r_active_log2) + 100.0 * torch.relu(_tag_sub_c_log2 - _tag_sub_c_active_log2)

    # 6. Tag H-tree Depth Constraint
    # DESTINY BankWithHtree.cpp sets numDataDistributeBit = assoc (number of ways to route).
    # It halves this count at every horizontal H-tree column level as it descends:
    #   level 0 (bank):  assoc / 2^log2(tag_bank_col_active)
    #   level 1 (mat):   ... / 2^log2(tag_sub_row_active)
    # If total horizontal levels > log2(assoc), ways_per_leaf < 1 -> mat declared invalid.
    # Constraint in log2-space: _tag_mat_r_enc + _tag_sub_r_active_log2 <= _assoc_enc
    tag_htree_depth_penalty = 100.0 * torch.relu(_tag_mat_r_enc + _tag_sub_r_active_log2 - _assoc_enc)
    
    return partition_penalty + rows_penalty + active_vs_total_penalty + tag_active_vs_total_penalty + tag_partition_penalty + data_cap_div_penalty + tag_cap_div_penalty + data_ww_div_penalty + tag_ww_div_penalty + tag_assoc_div_penalty + data_addr_penalty + tag_addr_penalty + tag_sub_active_vs_total_penalty + tag_htree_depth_penalty


def check_post_snap_partition(design, fixed_context):
    """
    Post-snap partition constraint check.
    Mirrors DESTINY main.cpp: blockSize / (active_mat_row * active_mat_col *
    active_sub_row * active_sub_col) == 0 causes silent discard of all configs.
    For normal/sequential cache access mode, blockSize == word_width_bits.
    """
    ww    = int(design.get("word_width_bits", fixed_context.get("word_width_bits", 64)))
    mat_r = int(design.get("data_num_active_mat_per_row", 1))
    mat_c = int(design.get("data_num_active_mat_per_col", 1))
    sub_r = int(design.get("data_num_active_subarray_per_row", 1))
    sub_c = int(design.get("data_num_active_subarray_per_col", 1))
    partition = mat_r * mat_c * sub_r * sub_c

    if partition > ww:
        print(
            f"  [warn] Post-snap partition constraint violated: "
            f"word_width={ww} < partition_product={partition} "
            f"(mat_r={mat_r}, mat_c={mat_c}, "
            f"sub_r={sub_r}, sub_c={sub_c}). "     
        )

    tag_mat_r = int(design.get("tag_num_active_mat_per_row", 1))
    tag_mat_c = int(design.get("tag_num_active_mat_per_col", 1))
    tag_sub_r = int(design.get("tag_num_active_subarray_per_row", 1))
    tag_sub_c = int(design.get("tag_num_active_subarray_per_col", 1))
    tag_partition = tag_mat_r * tag_mat_c * tag_sub_r * tag_sub_c
    tag_ww = 32

    if tag_partition > tag_ww:
        print(
            f"  [warn] Post-snap TAG partition constraint violated: "
            f"approx_tag_width={tag_ww} < tag_partition_product={tag_partition} "
        )
        
    assoc = int(design.get("associativity", fixed_context.get("associativity", 4)))
    tag_active_mats = tag_mat_r * tag_mat_c
    if tag_active_mats > assoc or assoc % tag_active_mats != 0:
        print(
            f"  [warn] Post-snap TAG associativity constraint violated: "
            f"assoc={assoc} not perfectly divisible by active_mats={tag_active_mats} "
        )
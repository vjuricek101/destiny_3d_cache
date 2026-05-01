import os
import re
from typing import Dict, List, Tuple, Optional

# ── Column Selection ───────────────────────────────────────────────────────────
# Columns kept in the final CSV.  Dropped columns fall into 3 categories:
#   1. Array-level PPA (data_read_latency_ns, data_bank_area_mm2, etc.) —
#      these are direct components of the cache-level targets; including them
#      would let the model trivially "add up" sub-components (target leakage).
#   2. Tag-array duplicates (tag_*) — the data array organization already
#      captures the structural information; tag details add noise for training.
#   3. Dimension/efficiency columns (bank_height_um, area_efficiency_pct, etc.)
#      — these are outputs of the layout solver, not structural design decisions,
#      and are highly correlated with the PPA targets we predict.
KEEP_COLS: List[str] = [
    # ── Variant identifier (added by Python, not DESTINY) ──────────────────
    "variant_name",
    # ── Swept / configuration inputs ──────────────────────────────────────
    "opt_target",
    "capacity_kb",
    "word_width_bits",
    "associativity",
    "temperature_K",
    "process_node_nm",
    "device_roadmap",
    "internal_sensing",
    "mem_cell_type",
    # ── Cache-level PPA targets ────────────────────────────────────────────
    "cache_access_mode",
    "cache_area_mm2",
    "cache_hit_latency_ns",
    "cache_miss_latency_ns",
    "cache_write_latency_ns",
    "cache_refresh_latency_ns",
    "cache_hit_energy_nJ",
    "cache_miss_energy_nJ",
    "cache_write_energy_nJ",
    "cache_refresh_energy_nJ",
    "cache_leakage_mW",
    "cache_refresh_power_W",
    # ── Structural / organizational outputs (data array only) ──────────────
    # These describe HOW DESTINY organized the memory — safe model features.
    "data_stacked_die_count",    # destiny_bank_stacked
    "data_total_banks",          # destiny_total_banks
    "data_total_mats",           # destiny_total_mats
    "data_num_row_subarray",     # destiny_mat_rows
    "data_num_col_subarray",     # destiny_mat_cols
    "data_subarray_num_row",     # destiny_subarray_rows
    "data_subarray_num_col",     # destiny_subarray_cols
    "data_mux_sense_amp",        # destiny_senseamp_mux
    "data_mux_output_lev2",      # destiny_output_mux_l2
    "data_num_active_mat_per_col",
    "data_num_active_mat_per_row",
    "data_num_active_subarray_per_col",
    "data_num_active_subarray_per_row",
    "data_num_row_per_set",
]

# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_cell_params(filepath: str) -> Dict[str, str]:
    """Lightweight parser for DESTINY .cell files."""
    params: Dict[str, str] = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'): continue
            if line.startswith('-'):
                parts = line[1:].split(':', 1)
                if len(parts) == 2:
                    params[parts[0].strip()] = parts[1].strip()
    return params

def extract_process_node(cell_filename: str) -> Optional[int]:
    m = re.search(r'_n(\d+)\.cell$', cell_filename)
    return int(m.group(1)) if m else None

def setup_dirs(mem_type: str, is_arch: bool = False) -> Tuple[str, str]:
    suffix = "_arch" if is_arch else ""
    temp_dir    = f"/dev/shm/vjuricek_destiny_tmp/{mem_type}{suffix}"
    results_dir = f"exploration_results/{mem_type}{suffix}"
    for d in [temp_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    return temp_dir, results_dir

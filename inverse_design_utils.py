import os
import re
import math
import subprocess
import threading
import shutil
import numpy as np
import pandas as pd
from destiny_utils import derive_sram_physical_params

BENCHING_LAYOUT_COLS = [
    "data_mux_sense_amp",
    "data_mux_output_lev1",
    "data_mux_output_lev2",
    "data_num_active_mat_per_row",
    "data_num_active_mat_per_col",
    "data_num_active_subarray_per_row",
    "data_num_active_subarray_per_col",
    "tag_num_row_mat",
    "tag_num_col_mat",
    "tag_mux_sense_amp",
    "tag_mux_output_lev1",
    "tag_mux_output_lev2",
    "tag_num_active_mat_per_row",
    "tag_num_active_mat_per_col",
    "tag_num_active_subarray_per_row",
    "tag_num_active_subarray_per_col",
    "tag_num_row_subarray",
    "tag_num_col_subarray",
    "tag_area_optimization_level",
    "tag_local_wire_type",
    "tag_local_wire_repeater_type",
    "tag_local_wire_low_swing",
    "tag_global_wire_type",
    "tag_global_wire_repeater_type",
    "tag_global_wire_low_swing",
    "data_local_wire_type",
    "data_local_wire_repeater_type",
    "data_local_wire_low_swing",
    "data_global_wire_type",
    "data_global_wire_repeater_type",
    "data_global_wire_low_swing",
    "data_area_optimization_level",
]

def map_repeater(val):
    """Normalize repeater name."""
    v = str(val)
    if "Fully-Optimized" in v or "Opt" in v: return "RepeatedOpt"
    if "No" in v: return "RepeatedNone"
    return v.replace(' ', '')

def map_wire_type(val):
    """Normalize wire type."""
    v = str(val).replace(' ', '')
    return v.replace('Semi-Global', 'Semi')

def generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage, cell_aspect_ratio, node, roadmap, opt_target, layout_config, free_mat=False, free_bank=False):
    """Writes physical cell parameters and array routing configuration files for DESTINY."""
    cell_params = {"SRAMCellNMOSWidth (F)": wn, "SRAMCellPMOSWidth (F)": wp, "AccessCMOSWidth (F)": wac}
    derive_sram_physical_params(cell_params, node)

    cell_content = f"""-MemCellType: SRAM
-CellArea (F^2): {cell_params["CellArea (F^2)"]:.5f}
-SRAMCellNMOSWidth (F): {wn:.4f}
-SRAMCellPMOSWidth (F): {wp:.4f}
-AccessCMOSWidth (F): {wac:.4f}
-AccessType: CMOS
-MinSenseVoltage (mV): {cell_params["MinSenseVoltage (mV)"]:.4f}
-CellAspectRatio: {cell_aspect_ratio:.4f}
-ReadVoltage (V): {read_voltage:.4f}
-Stitching: 16
"""

    cfg_content = f"""-OptimizationTarget: {opt_target}
-EnablePruning: Yes
-Capacity (KB): {cap_kb}
-WordWidth (bit): {ww}
-Associativity (for cache only): {assoc}
-StackedDieCount: {stack}
-Temperature (K): {temp}
-DeviceRoadmap: {roadmap}
-MemoryCellInputFile: {os.path.abspath(cell_file)}
-ProcessNode: {node}
"""

    if layout_config:
        if layout_config.get("data_mux_sense_amp") is not None:
            cfg_content += f"-ForceMuxSenseAmp: {int(layout_config['data_mux_sense_amp'])}\n"
        if layout_config.get("data_mux_output_lev1") is not None:
            cfg_content += f"-ForceMuxOutputLev1: {int(layout_config['data_mux_output_lev1'])}\n"
        if layout_config.get("data_mux_output_lev2") is not None:
            cfg_content += f"-ForceMuxOutputLev2: {int(layout_config['data_mux_output_lev2'])}\n"
        if layout_config.get("data_num_active_mat_per_col") is not None and layout_config.get("data_num_active_mat_per_row") is not None and not free_bank:
            A = C = int(layout_config["data_num_active_mat_per_col"])
            B = D = int(layout_config["data_num_active_mat_per_row"])
            cfg_content += f"-ForceBank (Total AxB, Active CxD): {A}x{B}, {C}x{D}\n"
            
        if layout_config.get("data_num_row_subarray") is not None and layout_config.get("data_num_col_subarray") is not None and not free_mat:
            A = int(layout_config["data_num_row_subarray"])
            B = int(layout_config["data_num_col_subarray"])
            C = int(layout_config.get("data_num_active_subarray_per_col", A))
            D = int(layout_config.get("data_num_active_subarray_per_row", B))
            A, B = max(A, C), max(B, D)
            cfg_content += f"-ForceMat (Total AxB, Active CxD): {A}x{B}, {C}x{D}\n"
        elif layout_config.get("data_num_active_subarray_per_col") is not None and layout_config.get("data_num_active_subarray_per_row") is not None and not free_mat:
            A = C = int(layout_config["data_num_active_subarray_per_col"])
            B = D = int(layout_config["data_num_active_subarray_per_row"])
            cfg_content += f"-ForceMat (Total AxB, Active CxD): {A}x{B}, {C}x{D}\n"
        
        if layout_config.get("tag_area_optimization_level") is not None:
            val = str(layout_config['tag_area_optimization_level']).lower()
            val = "latency" if "latency" in val else ("area" if "area" in val else layout_config['tag_area_optimization_level'])
            cfg_content += f"-TagBufferDesignOptimization: {val}\n"
        if layout_config.get("tag_local_wire_type") is not None:
            cfg_content += f"-TagLocalWireType: {map_wire_type(layout_config['tag_local_wire_type'])}\n"
        if layout_config.get("tag_local_wire_repeater_type") is not None:
            cfg_content += f"-TagLocalWireRepeaterType: {map_repeater(layout_config['tag_local_wire_repeater_type'])}\n"
        if layout_config.get("tag_local_wire_low_swing") is not None:
            cfg_content += f"-TagLocalWireUseLowSwing: {layout_config['tag_local_wire_low_swing']}\n"
        if layout_config.get("tag_global_wire_type") is not None:
            cfg_content += f"-TagGlobalWireType: {map_wire_type(layout_config['tag_global_wire_type'])}\n"
        if layout_config.get("tag_global_wire_repeater_type") is not None:
            cfg_content += f"-TagGlobalWireRepeaterType: {map_repeater(layout_config['tag_global_wire_repeater_type'])}\n"
        if layout_config.get("tag_global_wire_low_swing") is not None:
            cfg_content += f"-TagGlobalWireUseLowSwing: {layout_config['tag_global_wire_low_swing']}\n"
            
        if layout_config.get("tag_mux_output_lev1") is not None:
            cfg_content += f"-ForceTagMuxOutputLev1: {int(layout_config['tag_mux_output_lev1'])}\n"
        if layout_config.get("tag_mux_output_lev2") is not None:
            cfg_content += f"-ForceTagMuxOutputLev2: {int(layout_config['tag_mux_output_lev2'])}\n"
        
        if layout_config.get("tag_mux_sense_amp") is not None:
            cfg_content += f"-ForceTagMuxSenseAmp: {int(layout_config['tag_mux_sense_amp'])}\n"
        
        if layout_config.get("tag_num_row_mat") is not None and layout_config.get("tag_num_col_mat") is not None and not free_bank:
            A = int(layout_config["tag_num_row_mat"])
            B = int(layout_config["tag_num_col_mat"])
            C = int(layout_config.get("tag_num_active_mat_per_col", A))
            D = int(layout_config.get("tag_num_active_mat_per_row", B))
            A, B = max(A, C), max(B, D)
            cfg_content += f"-ForceTagBank (Total AxB, Active CxD): {A}x{B}, {C}x{D}\n"
        elif layout_config.get("tag_num_active_mat_per_col") is not None and layout_config.get("tag_num_active_mat_per_row") is not None and not free_bank:
            A = C = int(layout_config["tag_num_active_mat_per_col"])
            B = D = int(layout_config["tag_num_active_mat_per_row"])
            cfg_content += f"-ForceTagBank (Total AxB, Active CxD): {A}x{B}, {C}x{D}\n"
            
        if layout_config.get("tag_num_row_subarray") is not None and layout_config.get("tag_num_col_subarray") is not None and not free_mat:
            A = int(layout_config["tag_num_row_subarray"])
            B = int(layout_config["tag_num_col_subarray"])
            C = int(layout_config.get("tag_num_active_subarray_per_col", A))
            D = int(layout_config.get("tag_num_active_subarray_per_row", B))
            A, B = max(A, C), max(B, D)
            cfg_content += f"-ForceTagMat (Total AxB, Active CxD): {A}x{B}, {C}x{D}\n"
        elif layout_config.get("tag_num_active_subarray_per_col") is not None and layout_config.get("tag_num_active_subarray_per_row") is not None and not free_mat:
            A = C = int(layout_config["tag_num_active_subarray_per_col"])
            B = D = int(layout_config["tag_num_active_subarray_per_row"])
            cfg_content += f"-ForceTagMat (Total AxB, Active CxD): {A}x{B}, {C}x{D}\n"

        if layout_config.get("data_local_wire_type") is not None:
            cfg_content += f"-LocalWireType: {map_wire_type(layout_config['data_local_wire_type'])}\n"
        if layout_config.get("data_local_wire_repeater_type") is not None:
            cfg_content += f"-LocalWireRepeaterType: {map_repeater(layout_config['data_local_wire_repeater_type'])}\n"
        if layout_config.get("data_local_wire_low_swing") is not None:
            cfg_content += f"-LocalWireUseLowSwing: {layout_config['data_local_wire_low_swing']}\n"
        if layout_config.get("data_global_wire_type") is not None:
            cfg_content += f"-GlobalWireType: {map_wire_type(layout_config['data_global_wire_type'])}\n"
        if layout_config.get("data_global_wire_repeater_type") is not None:
            cfg_content += f"-GlobalWireRepeaterType: {map_repeater(layout_config['data_global_wire_repeater_type'])}\n"
        if layout_config.get("data_global_wire_low_swing") is not None:
            cfg_content += f"-GlobalWireUseLowSwing: {layout_config['data_global_wire_low_swing']}\n"

        if layout_config.get("data_area_optimization_level") is not None:
            val = str(layout_config['data_area_optimization_level']).lower()
            val = "latency" if "latency" in val else ("area" if "area" in val else layout_config['data_area_optimization_level'])
            cfg_content += f"-BufferDesignOptimization: {val}\n"

    with open(cell_file, "w") as f: f.write(cell_content)
    with open(cfg_file,  "w") as f: f.write(cfg_content)
    return cell_content, cfg_content

def run_destiny_process(cfg_file, timeout, verbose):
    """Launches DESTINY compiler process."""
    output_lines = []
    def _stream(pipe):
        for l in iter(pipe.readline, ""):
            output_lines.append(l.strip())
            if verbose: print(f"    [destiny] {l}", end="")

    process = subprocess.Popen(["./destiny", cfg_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    t = threading.Thread(target=_stream, args=(process.stdout,), daemon=True)
    t.start()
    t.join(timeout=timeout)

    if process.poll() is None:
        process.kill()
        print(f"  [warn] DESTINY validation timed out after {timeout}s.")
    return process, "\n".join(output_lines)

def parse_destiny_output(stdout_text, csv_file):
    """Extract PPA metrics from DESTINY stdout or CSV."""
    metrics_regex = {
        "cache_area_mm2": (
            r"(?:Total|Cache) Area\s*=\s*([\d.]+)\s*(mm\^2|um\^2)",
            lambda v, u: v / 1e6 if u == "um^2" else v,
        ),
        "cache_hit_latency_ns": (
            r"(?:Read|Cache Hit) Latency\s*=\s*([\d.]+)\s*(ps|ns|us)",
            lambda v, u: v / 1000 if u == "ps" else (v * 1000 if u == "us" else v),
        ),
        "cache_write_latency_ns": (
            r"(?:Write|Cache Write) Latency\s*=\s*([\d.]+)\s*(ps|ns|us)",
            lambda v, u: v / 1000 if u == "ps" else (v * 1000 if u == "us" else v),
        ),
        "cache_refresh_latency_ns": (
            r"(?:Refresh|Cache Refresh) Latency\s*=\s*([\d.]+)\s*(ps|ns|us)",
            lambda v, u: v / 1000 if u == "ps" else (v * 1000 if u == "us" else v),
        ),
        "cache_hit_energy_nJ": (
            r"(?:Read|Cache Hit) Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J",
            lambda v, u: v / 1000 if u == "p" else (v * 1000 if u == "u" else v),
        ),
        "cache_write_energy_nJ": (
            r"(?:Write|Cache Write) Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J",
            lambda v, u: v / 1000 if u == "p" else (v * 1000 if u == "u" else v),
        ),
        "cache_refresh_energy_nJ": (
            r"(?:Refresh|Cache Refresh) Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J",
            lambda v, u: v / 1000 if u == "p" else (v * 1000 if u == "u" else v),
        ),
        "cache_leakage_mW": (
            r"(?:Leakage|Cache Total Leakage) Power\s*=\s*([\d.]+)\s*(p|n|u|m)?W",
            lambda v, u: (
                v / 1000 if u == "u" else
                (v / 1e6 if u == "n" else
                (v / 1e9 if u == "p" else
                (v * 1000 if not u else v)))
            ),
        ),
    }
    destiny_ppa_metrics = {}
    for key, (regex, scale_fn) in metrics_regex.items():
        m = re.search(regex, stdout_text)
        if m:
            destiny_ppa_metrics[key] = scale_fn(float(m.group(1)), m.group(2) if m.lastindex >= 2 else "")
    
    if not destiny_ppa_metrics and os.path.exists(csv_file):
        ppa_df = pd.read_csv(csv_file)
        destiny_ppa_metrics = {c: float(ppa_df[c].iloc[0]) for c in metrics_regex if c in ppa_df.columns}
    return destiny_ppa_metrics or None

def validate_and_capture(cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage, cell_aspect_ratio,
                          node=32, roadmap="HP", timeout=60, verbose=False, opt_target="ReadLatency",
                          layout_config=None, prefix="validation_temp_bench"):
    """Compile layout with DESTINY and retrieve performance metrics under tier fallback options."""
    cell_file, cfg_file, csv_file = f"{prefix}.cell", f"{prefix}.cfg", f"{prefix}.csv"
    log_file = f"{prefix}.log"

    def _preflight_debug(tier_label, free_mat, free_bank):
        lc = layout_config or {}
        mat_r  = int(lc.get("data_num_active_mat_per_row",  1))
        mat_c  = int(lc.get("data_num_active_mat_per_col",  1))
        sub_r  = int(lc.get("data_num_active_subarray_per_row", 1))
        sub_c  = int(lc.get("data_num_active_subarray_per_col", 1))
        tot_r  = int(lc.get("data_num_row_mat",  mat_r))
        tot_c  = int(lc.get("data_num_col_mat",  mat_c))
        partition = mat_r * mat_c * sub_r * sub_c
        cap_bits  = cap_kb * 1024 * 8
        rps_denom = assoc * ww * mat_c * sub_c
        rps       = cap_bits / rps_denom if rps_denom else float("inf")
        ww_div_ok = (ww % partition == 0) if partition > 0 else False
        cap_div_ok = (cap_bits % (tot_r * tot_c) == 0) if tot_r * tot_c > 0 else False

        tag_mat_r = int(lc.get("tag_num_active_mat_per_row", 1))
        tag_mat_c = int(lc.get("tag_num_active_mat_per_col", 1))
        tag_part  = tag_mat_r * tag_mat_c
        tag_assoc_ok = (assoc % tag_part == 0) if tag_part > 0 else False

        print(f"  [dbg] {tier_label}: free_mat={free_mat}, free_bank={free_bank}")
        print(f"  [dbg]   Inputs : cap={cap_kb}KB  ww={ww}b  assoc={assoc}  stack={stack}  temp={temp}K  node={node}nm  roadmap={roadmap}")
        print(f"  [dbg]   Cell   : wn={wn:.4f}  wp={wp:.4f}  wac={wac:.4f}  rv={read_voltage:.4f}V  ar={cell_aspect_ratio:.4f}")
        if not free_mat:
            print(f"  [dbg]   DataMat: active=[{mat_r}x{mat_c}]  total=[{tot_r}x{tot_c}]  subarrays=[{sub_r}x{sub_c}]")
            print(f"  [dbg]   Partition product : {mat_r}*{mat_c}*{sub_r}*{sub_c} = {partition}  (ww={ww}) -> {'OK' if partition <= ww else 'VIOLATION: partition > ww'}")
            print(f"  [dbg]   WW divisibility   : ww={ww} % partition={partition} = {ww % partition if partition else 'N/A'} -> {'OK' if ww_div_ok else 'VIOLATION: not integer divisible'}")
            print(f"  [dbg]   Rows-per-subarray : cap_bits={cap_bits}/({assoc}*{ww}*{mat_c}*{sub_c}) = {rps:.4f} -> {'OK (integer)' if rps == int(rps) else 'VIOLATION: non-integer rps'} {'OK (>0)' if rps > 0 else 'VIOLATION: rps <= 0'}")
            print(f"  [dbg]   Cap/TotalMats div : cap_bits={cap_bits} / total_mats={tot_r * tot_c} = {cap_bits/(tot_r*tot_c) if tot_r*tot_c else 'N/A'} -> {'OK' if cap_div_ok else 'VIOLATION: non-integer'}")
        else:
            print(f"  [dbg]   DataMat: FREE (DESTINY auto-selects mat sizing)")
        if not free_bank:
            print(f"  [dbg]   Bank   : forced [{mat_c}x{mat_r}]")
        else:
            print(f"  [dbg]   Bank   : FREE")
        print(f"  [dbg]   TagMat : active=[{tag_mat_r}x{tag_mat_c}]  assoc={assoc} % tag_part={tag_part} -> {'OK' if tag_assoc_ok else 'VIOLATION: assoc not divisible by tag_part'}")

        tag_bank_col_active = int(lc.get("tag_num_active_mat_per_row", 1))
        tag_mat_col_active  = int(lc.get("tag_num_active_subarray_per_row", 1))
        tag_htree_horiz_levels = (math.log2(tag_bank_col_active) if tag_bank_col_active > 1 else 0) + \
                                  (math.log2(tag_mat_col_active)  if tag_mat_col_active  > 1 else 0)
        log2_assoc = math.log2(assoc) if assoc > 1 else 0
        ways_per_leaf = assoc / (2 ** tag_htree_horiz_levels) if tag_htree_horiz_levels >= 0 else assoc
        if not free_mat:
            tag_htree_ok = (tag_htree_horiz_levels <= log2_assoc) and (ways_per_leaf >= 1)
            print(f"  [dbg]   Tag H-tree: bank_col_active={tag_bank_col_active} mat_col_active={tag_mat_col_active} "
                  f"-> htree_horiz_levels={tag_htree_horiz_levels:.0f} vs log2(assoc)={log2_assoc:.0f} "
                  f"-> ways_per_leaf={ways_per_leaf:.2f} -> {'OK' if tag_htree_ok else 'VIOLATION: ways_per_leaf < 1 (tag bank will be invalid)'}")

    def _attempt(free_mat, free_bank, tier_label="tier?"):
        _preflight_debug(tier_label, free_mat, free_bank)
        generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp, wn, wp, wac,
                                  read_voltage, cell_aspect_ratio, node, roadmap, opt_target, layout_config,
                                  free_mat=free_mat, free_bank=free_bank)
        try:
            with open(cfg_file) as _f:
                _cfg_txt = _f.read()
            print(f"  [dbg] {tier_label} DESTINY cfg ({cfg_file}):\n" +
                  "\n".join(f"    {l}" for l in _cfg_txt.strip().splitlines()))
        except Exception as _e:
            print(f"  [dbg] {tier_label}: could not read cfg: {_e}")

        process, stdout_text = run_destiny_process(cfg_file, timeout, verbose)
        with open(log_file, "w") as f:
            f.write(stdout_text)

        csv_exists     = os.path.exists(csv_file)
        has_1e50       = "1e+50" in stdout_text
        has_nosol      = "numSolutions = 0" in stdout_text
        bad_retcode    = (process.returncode != 0 and process.returncode is not None)
        success_signal = "Finished!" in stdout_text or "csv generated successfully" in stdout_text or csv_exists

        if bad_retcode or has_1e50 or has_nosol or not success_signal:
            reasons = []
            if bad_retcode:    reasons.append(f"non-zero returncode={process.returncode}")
            if has_1e50:       reasons.append("sentinel value 1e+50 in output")
            if has_nosol:      reasons.append("numSolutions=0")
            if not csv_exists: reasons.append("no CSV output")
            if not success_signal: reasons.append("no success signal")
            error_lines = [l.strip() for l in stdout_text.splitlines()
                           if any(kw in l for kw in ["numSolutions", "numDesigns", "No valid", "Error", "error", "WARNING"])]
            print(f"  [dbg] {tier_label} FAILED. Reasons: {'; '.join(reasons)}")
            if error_lines:
                print(f"  [dbg]   DESTINY diagnostic lines:")
                for el in error_lines:
                    print(f"    > {el}")
            return None

        parsed = parse_destiny_output(stdout_text, csv_file)
        if parsed is None:
            print(f"  [dbg] {tier_label} FAILED. Parse returned None")
            return None
        print(f"  [dbg] {tier_label} PASSED.")
        return parsed

    def _archive(suffix):
        for ext in (".cell", ".cfg", ".log"):
            src = f"{prefix}{ext}"
            if os.path.exists(src):
                shutil.copy(src, f"{prefix}_{suffix}{ext}")

    result = _attempt(free_mat=False, free_bank=False, tier_label="tier1")
    if result is not None:
        return result

    print(f"  [warn] DESTINY tier-1 failed; retrying with free mat (tier 2).")
    _archive("tier1")
    result = _attempt(free_mat=True, free_bank=False, tier_label="tier2")
    if result is not None:
        result["_used_fallback_mat"] = True
        return result

    print(f"  [warn] DESTINY tier-2 failed; retrying with free bank (tier 3).")
    _archive("tier2")
    result = _attempt(free_mat=False, free_bank=True, tier_label="tier3")
    if result is not None:
        result["_used_fallback_bank"] = True
        return result

    print(f"  [warn] DESTINY tier-3 failed; retrying with free bank+mat (tier 4).")
    _archive("tier3")
    result = _attempt(free_mat=True, free_bank=True, tier_label="tier4")
    if result is not None:
        result["_used_fallback_mat"] = True
        result["_used_fallback_bank"] = True
    return result

def row_to_context(row, roadmap):
    """Convert a dataset row to optimizer context format."""
    node = int(row["process_node_nm"])
    ctx  = {f"process_node_nm_{node}": 1.0, "temperature_K": float(row.get("temperature_K", 350.0))}
    for rm in ["HP", "LOP", "LSTP"]: ctx[f"device_roadmap_{rm}"] = 1.0 if rm == roadmap else 0.0
    ctx["_wn"]   = float(row.get("CellInput_SRAMCellNMOSWidth (F)", 2.5))
    ctx["_wp"]   = float(row.get("CellInput_SRAMCellPMOSWidth (F)", 2.0))
    ctx["_wac"]  = float(row.get("CellInput_AccessCMOSWidth (F)",   2.5))
    ctx["_read_voltage"] = float(row.get("CellInput_ReadVoltage (V)", 1.0))
    ctx["_cell_aspect_ratio"] = float(row.get("CellInput_CellAspectRatio", 1.4600))
    ctx["_temp"] = int(row.get("temperature_K", 350))
    if hasattr(row, "index") and "opt_target" in row.index: ctx["_opt_target"] = str(row["opt_target"])
    return ctx

def pct_err(predicted, target):
    """Return percentage error."""
    return float("nan") if target == 0 or predicted is None or np.isnan(predicted) else (predicted - target) / abs(target) * 100.0

def layout_from_row(row, prefix=""):
    """Extract layout parameters from a row."""
    return {k: row[prefix+k] for k in BENCHING_LAYOUT_COLS if prefix+k in row.index and not (isinstance(row[prefix+k], float) and np.isnan(row[prefix+k]))}

#!/usr/bin/env python3
"""Runs objective based sweep optimization across baseline designs + produces CSVs + plots"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys, argparse, subprocess, warnings, time, threading, shutil, re, glob, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from inverse_design_gumbel_objective import InverseOptimizerGumbel
from destiny_utils import (
    pareto_frontier_2d,
    pareto_frontier_nd,
    pareto_step_line,
    cap_colormap,
    format_log_axis,
    add_cap_colorbar,
    derive_sram_physical_params,
    setup_opt_dirs,
    validate_cache_geometry,
    METRIC_META,
    METRIC_TO_PPA_LABEL,
    LAYOUT_COLS as _LAYOUT_COLS,
    TARGET_COLS as TARGET_KEYS,
    TECH_SKIP_TARGETS,
    get_active_targets,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Temperature co-varies with stack count (from run_exploration.py).
SRAM_TEMPERATURE_MAP = {1: 300, 2: 363, 4: 380}

def _map_repeater(val):
    v = str(val)
    if "Fully-Optimized" in v or "Opt" in v: return "RepeatedOpt"
    if "No" in v: return "RepeatedNone"
    return v.replace(' ', '')

def _map_wire_type(val):
    v = str(val).replace(' ', '')
    v = v.replace('Semi-Global', 'Semi')
    return v

def _generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage, cell_aspect_ratio, node, roadmap, opt_target, layout_config, free_mat=False, free_bank=False):
    """Writes physical cell parameters and array routing configuration files to invoke DESTINY compiler."""
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

    # Forces architectural matrix sizing and multiplexer routing configurations solved by backpropagation
    if layout_config:
        print("LAYOUT CONFIG:", layout_config)
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
            cfg_content += f"-TagLocalWireType: {_map_wire_type(layout_config['tag_local_wire_type'])}\n"
        if layout_config.get("tag_local_wire_repeater_type") is not None:
            cfg_content += f"-TagLocalWireRepeaterType: {_map_repeater(layout_config['tag_local_wire_repeater_type'])}\n"
        if layout_config.get("tag_local_wire_low_swing") is not None:
            cfg_content += f"-TagLocalWireUseLowSwing: {layout_config['tag_local_wire_low_swing']}\n"
        if layout_config.get("tag_global_wire_type") is not None:
            cfg_content += f"-TagGlobalWireType: {_map_wire_type(layout_config['tag_global_wire_type'])}\n"
        if layout_config.get("tag_global_wire_repeater_type") is not None:
            cfg_content += f"-TagGlobalWireRepeaterType: {_map_repeater(layout_config['tag_global_wire_repeater_type'])}\n"
        if layout_config.get("tag_global_wire_low_swing") is not None:
            cfg_content += f"-TagGlobalWireUseLowSwing: {layout_config['tag_global_wire_low_swing']}\n"
            
        if layout_config.get("tag_mux_output_lev1") is not None:
            cfg_content += f"-ForceTagMuxOutputLev1: {int(layout_config['tag_mux_output_lev1'])}\n"
        if layout_config.get("tag_mux_output_lev2") is not None:
            cfg_content += f"-ForceTagMuxOutputLev2: {int(layout_config['tag_mux_output_lev2'])}\n"
        
        # for subarray layout forcing
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

        # Array wiring configurations
        if layout_config.get("data_local_wire_type") is not None:
            cfg_content += f"-LocalWireType: {_map_wire_type(layout_config['data_local_wire_type'])}\n"
        if layout_config.get("data_local_wire_repeater_type") is not None:
            cfg_content += f"-LocalWireRepeaterType: {_map_repeater(layout_config['data_local_wire_repeater_type'])}\n"
        if layout_config.get("data_local_wire_low_swing") is not None:
            cfg_content += f"-LocalWireUseLowSwing: {layout_config['data_local_wire_low_swing']}\n"
        if layout_config.get("data_global_wire_type") is not None:
            cfg_content += f"-GlobalWireType: {_map_wire_type(layout_config['data_global_wire_type'])}\n"
        if layout_config.get("data_global_wire_repeater_type") is not None:
            cfg_content += f"-GlobalWireRepeaterType: {_map_repeater(layout_config['data_global_wire_repeater_type'])}\n"
        if layout_config.get("data_global_wire_low_swing") is not None:
            cfg_content += f"-GlobalWireUseLowSwing: {layout_config['data_global_wire_low_swing']}\n"

        # Buffer design optimization level
        if layout_config.get("data_area_optimization_level") is not None:
            val = str(layout_config['data_area_optimization_level']).lower()
            val = "latency" if "latency" in val else ("area" if "area" in val else layout_config['data_area_optimization_level'])
            cfg_content += f"-BufferDesignOptimization: {val}\n"

    with open(cell_file, "w") as f: f.write(cell_content)
    with open(cfg_file,  "w") as f: f.write(cfg_content)
    return cell_content, cfg_content

def _run_destiny_process(cfg_file, timeout, verbose):
    """Launches DESTINY"""
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

def _parse_destiny_output(stdout_text, csv_file):
    """Extracts cache-level physical latency, area, energy, and leakage metrics from simulator logs."""
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

def validate_and_capture(tech, cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage, cell_aspect_ratio,
                          node=32, roadmap="HP", timeout=60, verbose=False, opt_target="ReadLatency",
                          layout_config=None, prefix="validation_temp_bench"):
    """Orchestrates layout generation, simulator execution, and physical parameter extraction."""
    cell_file, cfg_file, csv_file = f"{prefix}.cell", f"{prefix}.cfg", f"{prefix}.csv"
    log_file = f"{prefix}.log"

    def _preflight_debug(tier_label, free_mat, free_bank):
        """Log constraint pre-flight checks before invoking DESTINY."""
        lc = layout_config or {}
        mat_r  = int(lc.get("data_num_active_mat_per_row",  1))
        mat_c  = int(lc.get("data_num_active_mat_per_col",  1))
        sub_r  = int(lc.get("data_num_active_subarray_per_row", 1))
        sub_c  = int(lc.get("data_num_active_subarray_per_col", 1))
        tot_r  = int(lc.get("data_num_row_mat",  mat_r))
        tot_c  = int(lc.get("data_num_col_mat",  mat_c))
        partition = mat_r * mat_c * sub_r * sub_c
        cap_bits  = cap_kb * 1024 * 8
        # rows-per-subarray (DESTINY BankWithHtree.cpp:381)
        rps_denom = assoc * ww * mat_c * sub_c
        rps       = cap_bits / rps_denom if rps_denom else float("inf")
        # word-width divisibility by active partitions
        ww_div_ok = (ww % partition == 0) if partition > 0 else False
        # capacity divisibility by total mats
        total_mats = tot_r * tot_c
        cap_div_ok = (cap_bits % total_mats == 0) if total_mats > 0 else False

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
            print(f"  [dbg]   Cap/TotalMats div : cap_bits={cap_bits} / total_mats={total_mats} = {cap_bits/total_mats if total_mats else 'N/A'} -> {'OK' if cap_div_ok else 'VIOLATION: non-integer'}")
        else:
            print(f"  [dbg]   DataMat: FREE (DESTINY auto-selects mat sizing)")
        if not free_bank:
            b_tot_r = int(lc.get("data_num_active_mat_per_col", mat_c))
            b_tot_c = int(lc.get("data_num_active_mat_per_row", mat_r))
            print(f"  [dbg]   Bank   : forced [{b_tot_r}x{b_tot_c}]")
        else:
            print(f"  [dbg]   Bank   : FREE")
        print(f"  [dbg]   TagMat : active=[{tag_mat_r}x{tag_mat_c}]  assoc={assoc} % tag_part={tag_part} -> {'OK' if tag_assoc_ok else 'VIOLATION: assoc not divisible by tag_part'}")

        # Tag H-tree depth check (BankWithHtree.cpp: numDataDistributeBit = assoc, halved at each horizontal level)
        # Tag bank col active = tag_num_active_mat_per_row (the column dimension of the tag bank)
        # Tag mat col active = tag_num_active_subarray_per_row
        lc = layout_config or {}
        tag_bank_col_active = int(lc.get("tag_num_active_mat_per_row", 1))
        tag_mat_col_active  = int(lc.get("tag_num_active_subarray_per_row", 1))
        import math
        tag_htree_horiz_levels = (math.log2(tag_bank_col_active) if tag_bank_col_active > 1 else 0) + \
                                  (math.log2(tag_mat_col_active)  if tag_mat_col_active  > 1 else 0)
        log2_assoc = math.log2(assoc) if assoc > 1 else 0
        # After tag_htree_horiz_levels halvings: ways_per_leaf = assoc / 2^tag_htree_horiz_levels
        ways_per_leaf = assoc / (2 ** tag_htree_horiz_levels) if tag_htree_horiz_levels >= 0 else assoc
        if not free_mat:
            tag_htree_ok = (tag_htree_horiz_levels <= log2_assoc) and (ways_per_leaf >= 1)
            print(f"  [dbg]   Tag H-tree: bank_col_active={tag_bank_col_active} mat_col_active={tag_mat_col_active} "
                  f"-> htree_horiz_levels={tag_htree_horiz_levels:.0f} vs log2(assoc)={log2_assoc:.0f} "
                  f"-> ways_per_leaf={ways_per_leaf:.2f} -> {'OK' if tag_htree_ok else 'VIOLATION: ways_per_leaf < 1 (tag bank will be invalid)'}")

    def _attempt(free_mat, free_bank, tier_label="tier?"):
        """Run DESTINY once; returns parsed metrics dict or None on failure."""
        _preflight_debug(tier_label, free_mat, free_bank)
        _generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp, wn, wp, wac,
                                  read_voltage, cell_aspect_ratio, node, roadmap, opt_target, layout_config,
                                  free_mat=free_mat, free_bank=free_bank)
        # Echo the config file that will be sent to DESTINY
        try:
            with open(cfg_file) as _f:
                _cfg_txt = _f.read()
            print(f"  [dbg] {tier_label} DESTINY cfg ({cfg_file}):\n" +
                  "\n".join(f"    {l}" for l in _cfg_txt.strip().splitlines()))
        except Exception as _e:
            print(f"  [dbg] {tier_label}: could not read cfg: {_e}")

        process, stdout_text = _run_destiny_process(cfg_file, timeout, verbose)
        with open(log_file, "w") as f:
            f.write(stdout_text)

        # Diagnose failure signals
        csv_exists     = os.path.exists(csv_file)
        has_1e50       = "1e+50" in stdout_text
        has_nosol      = "numSolutions = 0" in stdout_text
        bad_retcode    = (process.returncode != 0 and process.returncode is not None)
        success_signal = "Finished!" in stdout_text or "csv generated successfully" in stdout_text or csv_exists

        if bad_retcode or has_1e50 or has_nosol or not success_signal:
            reasons = []
            if bad_retcode:    reasons.append(f"non-zero returncode={process.returncode}")
            if has_1e50:       reasons.append("sentinel value 1e+50 in output")
            if has_nosol:      reasons.append("numSolutions=0 (partition/geometry constraint violated inside DESTINY)")
            if not csv_exists: reasons.append("no CSV output produced")
            if not success_signal: reasons.append("no 'Finished!' or 'csv generated successfully' signal")
            # Extract any DESTINY error lines for context
            error_lines = [l.strip() for l in stdout_text.splitlines()
                           if any(kw in l for kw in ["numSolutions", "numDesigns", "No valid", "Error", "error", "WARNING", "numAddressBit", "blockSize"])]
            print(f"  [dbg] {tier_label} FAILED. Reasons: {'; '.join(reasons)}")
            if error_lines:
                print(f"  [dbg]   DESTINY diagnostic lines:")
                for el in error_lines:
                    print(f"    > {el}")
            return None

        parsed = _parse_destiny_output(stdout_text, csv_file)
        if parsed is None:
            print(f"  [dbg] {tier_label} FAILED. Reasons: DESTINY ran but _parse_destiny_output returned None (no metrics extracted from stdout or CSV)")
            return None
        print(f"  [dbg] {tier_label} PASSED.")
        return parsed

    def _archive(suffix):
        """Copy current cell/cfg/log to a suffixed backup before overwriting."""
        for ext in (".cell", ".cfg", ".log"):
            src = f"{prefix}{ext}"
            if os.path.exists(src):
                shutil.copy(src, f"{prefix}_{suffix}{ext}")

    # Tier 1: force everything (bank + mat)
    result = _attempt(free_mat=False, free_bank=False, tier_label="tier1")
    if result is not None:
        return result

    # Tier 2: free mat, keep bank forced
    print(f"  [warn] DESTINY tier-1 (forced bank+mat) failed; retrying with free mat (tier 2).")
    _archive("tier1")
    result = _attempt(free_mat=True, free_bank=False, tier_label="tier2")
    if result is not None:
        result["_used_fallback_mat"] = True
        return result

    # Tier 3: free bank, keep mat forced
    print(f"  [warn] DESTINY tier-2 (free mat) failed; retrying with free bank (tier 3).")
    _archive("tier2")
    result = _attempt(free_mat=False, free_bank=True, tier_label="tier3")
    if result is not None:
        result["_used_fallback_bank"] = True
        return result

    # Tier 4: free both bank and mat
    print(f"  [warn] DESTINY tier-3 (free bank) failed; retrying with free bank+mat (tier 4).")
    _archive("tier3")
    result = _attempt(free_mat=True, free_bank=True, tier_label="tier4")
    if result is not None:
        result["_used_fallback_mat"] = True
        result["_used_fallback_bank"] = True
    return result

def row_to_context(row, roadmap):
    """Constructs active neural network routing context based on the process node constraint."""
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
    """Calculates the percentage relative change between optimized surrogate predictions and baseline metrics."""
    return float("nan") if target == 0 or predicted is None or np.isnan(predicted) else (predicted - target) / abs(target) * 100.0

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

def _layout_from_row(row, prefix=""):
    """Maps layout matrix configurations directly to physical array forced routing definitions."""
    return {k: row[prefix+k] for k in BENCHING_LAYOUT_COLS if prefix+k in row.index and not (isinstance(row[prefix+k], float) and np.isnan(row[prefix+k]))}

def _load_and_filter_data(args, data_csv):
    """Applies operating parameter constraints to exclude numerical compiler crashes and invalid layouts."""
    print(f"[1/5] Loading data sweep library: {data_csv}")
    ppa_df = pd.read_csv(data_csv)
    # Hardware bounds filter extremely non-optimal sizing coordinates and simulator sentinels:
    ppa_df = ppa_df[
        (ppa_df["mem_cell_type"].str.upper() == args.tech.upper()) &
        (ppa_df["process_node_nm"] == args.node) &
        (ppa_df["device_roadmap"].str.upper() == args.roadmap.upper()) &
        (ppa_df["cache_hit_latency_ns"] < 100) & (ppa_df["cache_area_mm2"] < 1000) &
        (ppa_df["cache_write_energy_nJ"]  < 1000) & (ppa_df["cache_leakage_mW"] > 0) &
        (ppa_df["cache_leakage_mW"] < 1e7)
    ]
    if getattr(args, "capacity_kb", None) is not None:
        ppa_df = ppa_df[ppa_df["capacity_kb"] == args.capacity_kb]
    if getattr(args, "associativity", None) is not None:
        ppa_df = ppa_df[ppa_df["associativity"] == args.associativity]
    if getattr(args, "word_width_bits", None) is not None:
        ppa_df = ppa_df[ppa_df["word_width_bits"] == args.word_width_bits]

    # Positivity filter only applies to metrics that are physically non-zero for this tech.
    active = set(get_active_targets(args.tech))
    for m in args.metrics:
        if m in active:
            ppa_df = ppa_df[ppa_df[m] > 0]
    print(f"   Library size after screening: {len(ppa_df):,}")
    if len(ppa_df) == 0: sys.exit("ERROR: Filter returned 0 matching target rows.")
    return ppa_df

def _run_optimization_sweep(target_vectors, ppa_data_frame, sizing_optimizer, args):
    """Solves sizing variables generically across any subset of the PPA objectives."""
    from train_model import CATEGORICAL_COLS
    ppa_results_records = []
    for idx, (i, row) in enumerate(target_vectors.iterrows()):
        ctx = row_to_context(row, args.roadmap)
        opt_ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}
        opt_ctx["data_stacked_die_count"] = 1.0 # Force 2D planar SRAM configurations
        
        # Pin the specific cache specifications of the baseline row being optimized
        opt_ctx["capacity_kb"] = float(row["capacity_kb"])
        opt_ctx["associativity"] = float(row["associativity"])
        if "word_width_bits" in row:
            opt_ctx["word_width_bits"] = float(row["word_width_bits"])
        
        # Ensure the baseline cache geometry is physically valid before optimizing
        validate_cache_geometry(
            capacity_kb=float(row["capacity_kb"]),
            associativity=int(row["associativity"]),
            word_width_bits=int(row["word_width_bits"]) if "word_width_bits" in row and not pd.isna(row["word_width_bits"]) else None
        )
        
        # Populate one-hot encoded wire, buffer, and cell categoricals from the target row
        for col in sizing_optimizer.feature_cols:
            for cat in CATEGORICAL_COLS:
                if cat not in ["process_node_nm", "device_roadmap", "mem_cell_type"] and col.startswith(f"{cat}_"):
                    val = str(row.get(cat, ""))
                    opt_ctx[col] = 1.0 if col == f"{cat}_{val}" else 0.0
        
        snapped_design_params, surrogate_ppa_predictions, pre_snap, snapped_surrogate_ppa = sizing_optimizer.optimize(
            args.metrics, opt_ctx, steps=args.opt_steps, verbose=args.verbose_opt
        )
        
        rec = {
            "target_idx": i, "node_nm": int(row["process_node_nm"]), "device_roadmap": args.roadmap, "method": "gumbel",
            "capacity_kb": snapped_design_params.get("capacity_kb", row.get("capacity_kb")), "word_width_bits": snapped_design_params.get("word_width_bits", row.get("word_width_bits")), "associativity": snapped_design_params.get("associativity", row.get("associativity")), "data_stacked_die_count": snapped_design_params.get("data_stacked_die_count", row.get("data_stacked_die_count")),
            **{k: snapped_design_params.get(k, row.get(k)) for k in BENCHING_LAYOUT_COLS},
            "opt_wn": snapped_design_params.get("CellInput_SRAMCellNMOSWidth (F)", ctx["_wn"]), "opt_wp": snapped_design_params.get("CellInput_SRAMCellPMOSWidth (F)", ctx["_wp"]), "opt_wac": snapped_design_params.get("CellInput_AccessCMOSWidth (F)", ctx["_wac"]), "opt_rv": snapped_design_params.get("CellInput_ReadVoltage (V)", ctx["_read_voltage"]), "opt_ar": snapped_design_params.get("CellInput_CellAspectRatio", ctx["_cell_aspect_ratio"]),
            "_wn": snapped_design_params.get("CellInput_SRAMCellNMOSWidth (F)", ctx["_wn"]), "_wp": snapped_design_params.get("CellInput_SRAMCellPMOSWidth (F)", ctx["_wp"]), "_wac": snapped_design_params.get("CellInput_AccessCMOSWidth (F)", ctx["_wac"]), "_read_voltage": snapped_design_params.get("CellInput_ReadVoltage (V)", ctx["_read_voltage"]), "_cell_aspect_ratio": snapped_design_params.get("CellInput_CellAspectRatio", ctx["_cell_aspect_ratio"]),
            "_temp": SRAM_TEMPERATURE_MAP.get(snapped_design_params.get("data_stacked_die_count", 1), ctx["_temp"]),
            "variant_name": row.get("variant_name"),
        }
        
        for k in args.metrics:
            label = METRIC_TO_PPA_LABEL.get(k, "")
            
            s_val = surrogate_ppa_predictions.get(label) if surrogate_ppa_predictions else None
            rec[f"surr_{k}"] = s_val
            
            snap_val = snapped_surrogate_ppa.get(label) if snapped_surrogate_ppa else None
            rec[f"post_snap_surr_{k}"] = snap_val
 
        for k in args.metrics:
            rec[f"destiny_{k}"] = None
        rec["destiny_used_fallback_mat"] = None
        rec["destiny_used_fallback_bank"] = None
        rec["destiny_used_fallback"] = None
 
        for k, v in pre_snap.items(): rec[f"pre_snap_{k}"] = v
        ppa_results_records.append(rec)
        
        print(f"  [{idx+1:3d}/{len(target_vectors)}] Sized Pareto Point #{idx} (Cap: {row.get('capacity_kb')} KB, Assoc: {row.get('associativity')}, WW: {row.get('word_width_bits')}):")
        for k in args.metrics:
            label = METRIC_TO_PPA_LABEL.get(k, "")
            s_val = surrogate_ppa_predictions.get(label) if surrogate_ppa_predictions else None
            snap_val = snapped_surrogate_ppa.get(label) if snapped_surrogate_ppa else None
            print(f"         {k}: Pre-snap={s_val:.4g}, Post-snap={snap_val:.4g}")
              
    return pd.DataFrame(ppa_results_records)

def _validate_top_designs(ppa_results, n_validate, args, work_dir):
    """Executes physical verification loop for top layout solutions."""
    if n_validate <= 0:
        print("\n[4/5] Skipping physical validator DESTINY validation runs (--validate-top 0)\n")
        return ppa_results

    print(f"\n[4/5] Running physical validation for top-{n_validate} optimal configurations...\n")
    for rank, ri in enumerate(ppa_results.index[:n_validate], 1):
        row = ppa_results.loc[ri]
        
        print(f"  [rank {rank}] variant_name={row.get('variant_name', 'N/A')}")
        
        # 1. Run validation on the optimized configuration
        layout_config = _layout_from_row(row)
        opt_prefix = os.path.join(work_dir, f"rank{rank}_opt")
        destiny_ppa = validate_and_capture(
            tech=args.tech, cap_kb=int(row.capacity_kb), ww=int(row.word_width_bits),
            assoc=int(row.associativity), stack=max(1, int(row.data_stacked_die_count)),
            temp=int(row["_temp"]), wn=float(row["_wn"]), wp=float(row["_wp"]), wac=float(row["_wac"]), read_voltage=float(row["_read_voltage"]), cell_aspect_ratio=float(row["_cell_aspect_ratio"]),
            node=int(row.get("node_nm", args.node)), roadmap=args.roadmap, timeout=args.destiny_timeout, verbose=args.verbose_destiny, opt_target="ReadLatency",
            layout_config=layout_config, prefix=opt_prefix
        )
        if destiny_ppa is not None:
            fb_mat = destiny_ppa.get("_used_fallback_mat", False)
            fb_bank = destiny_ppa.get("_used_fallback_bank", False)
            ppa_results.at[ri, "destiny_used_fallback_mat"] = fb_mat
            ppa_results.at[ri, "destiny_used_fallback_bank"] = fb_bank
            ppa_results.at[ri, "destiny_used_fallback"] = fb_mat or fb_bank

            print(f"    Validation Solver PPA Extracted:")
            for k in args.metrics:
                dx = destiny_ppa.get(k)
                ppa_results.at[ri, f"destiny_{k}"] = dx
                print(f"         {k}: Physical Destiny={dx:.4g}")
            print()
        else:
            print("    [warn] Validation failed.\n")
            
    return ppa_results

def _plot_results_side_effect(ppa_results, ppa_data_frame, target_vectors, args, plots_dir):
    """Single-panel plot:
    2D scatter of first two metrics displaying the library background, the Pareto frontier step-line,
    and the optimized design points (both post-snap surrogate and destiny physical validation),
    filtered by the capacities actually optimized in the sweep.
    """
    if len(args.metrics) < 2:
        return

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    x_col, y_col = args.metrics[0], args.metrics[1]
    
    unique_caps = target_vectors["capacity_kb"].unique()

    # Load and accumulate background files for color normalization
    all_bg_dfs = []
    for c_val in sorted(unique_caps):
        cap_str = str(int(c_val))
        cap_csv = os.path.join("pareto", args.tech, f"{args.tech}_cap_{cap_str}_pareto.csv")
        if os.path.exists(cap_csv):
            df_cap = pd.read_csv(cap_csv)
            # Filter to ensure valid metrics
            df_cap = df_cap[(df_cap[x_col] > 0) & (df_cap[y_col] > 0)]
            all_bg_dfs.append(df_cap)
        else:
            bg_cap = ppa_data_frame[ppa_data_frame["capacity_kb"] == c_val]
            all_bg_dfs.append(bg_cap)

    if all_bg_dfs:
        combined_bg = pd.concat(all_bg_dfs, ignore_index=True)
    else:
        combined_bg = ppa_data_frame

    uniq_ww = sorted(combined_bg["word_width_bits"].unique())
    norm = LogNorm(vmin=min(uniq_ww), vmax=max(uniq_ww))

    # Plot each capacity's background and Pareto frontier
    for c_val in sorted(unique_caps):
        cap_str = str(int(c_val))
        cap_csv = os.path.join("pareto", args.tech, f"{args.tech}_cap_{cap_str}_pareto.csv")
        if os.path.exists(cap_csv):
            df_cap = pd.read_csv(cap_csv)
            df_cap = df_cap[(df_cap[x_col] > 0) & (df_cap[y_col] > 0)]
        else:
            df_cap = ppa_data_frame[ppa_data_frame["capacity_kb"] == c_val]

        x_bg = df_cap[x_col].values
        y_bg = df_cap[y_col].values

        if len(x_bg) > 0:
            # 1. Plot Library Background/Pareto configurations for this capacity colored by word width
            ax.scatter(
                df_cap[x_col], df_cap[y_col],
                c=df_cap["word_width_bits"], norm=norm, cmap=plt.cm.viridis,
                s=15, alpha=0.3, linewidths=0, zorder=1,
                label=f"Library Pareto ({c_val} KB)"
            )

    # 3. Plot Optimized Designs (Post-Snap Surrogate)
    mask_snap = (
        ppa_results[f"post_snap_surr_{x_col}"].notna()
        & ppa_results[f"post_snap_surr_{y_col}"].notna()
    )
    if mask_snap.any():
        ax.scatter(
            ppa_results.loc[mask_snap, f"post_snap_surr_{x_col}"],
            ppa_results.loc[mask_snap, f"post_snap_surr_{y_col}"],
            c="#ffa657", s=60, marker="x", zorder=3, label="Optimized (Surrogate)"
        )

    # 4. Plot Optimized Designs (DESTINY Physical)
    mask_dest = (
        ppa_results[f"destiny_{x_col}"].notna()
        & ppa_results[f"destiny_{y_col}"].notna()
    )
    if mask_dest.any():
        ax.scatter(
            ppa_results.loc[mask_dest, f"destiny_{x_col}"],
            ppa_results.loc[mask_dest, f"destiny_{y_col}"],
            c="#3fb950", s=80, marker="*", edgecolors="k",
            linewidths=0.5, zorder=4, label="Optimized (DESTINY)"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(METRIC_META[x_col]["label"])
    ax.set_ylabel(METRIC_META[y_col]["label"])
    format_log_axis(ax, axis="both")
    ax.set_title(f"Objective Minimization Optimization: {x_col} vs {y_col}", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=norm), ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Word Width (bits)", fontsize=9)
    cbar.set_ticks(uniq_ww)
    cbar.set_ticklabels([f"{int(w)}b" for w in uniq_ww])

    fig.suptitle(
        f"{args.tech} | {args.node}nm | {args.roadmap} | method=gumbel (Direct Minimization)",
        fontsize=12, fontweight="bold"
    )
    
    plots_path = os.path.join(
        plots_dir,
        f"benchmark_trajectory_{args.tech}_{args.node}nm_{args.roadmap}_gumbel.png"
    )
    fig.savefig(plots_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plots_path}")

def main():
    p = argparse.ArgumentParser(description="Multi-metric cache objective inverse sizing benchmark engine.")
    p.add_argument("--tech", default="SRAM")
    p.add_argument("--node", type=int, default=32, help="Process node nm constraint")
    p.add_argument("--roadmap", default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--targets", nargs="+", default=["all"], help="PPA objectives to minimize (default: all active metrics for --tech)")
    p.add_argument("--opt-steps", type=int, default=400, help="Sizing optimization gradient steps")
    p.add_argument("--validate-top", type=int, default=5, help="Number of solved layouts to validate with C++ compiler")
    p.add_argument("--destiny-timeout", type=int, default=60, help="TIMEOUT for physical compilation subprocesses")
    
    # Input parameter flags
    p.add_argument("--capacity-kb", type=float, default=None, help="Filter the target vectors to a specific capacity (in KB)")
    p.add_argument("--associativity", type=int, default=None, help="Filter the target vectors to a specific associativity")
    p.add_argument("--word-width-bits", type=int, default=None, help="Filter the target vectors to a specific word width")
    
    p.add_argument("--run-id", default=None, help="Optional run ID for locating the output directory under opt_obj/")
    p.add_argument("--verbose-destiny", action="store_true", help="Print compiler outputs directly to console")
    p.add_argument("--verbose-opt", action="store_true", help="Print tabular design updates during steps")
    args = p.parse_args()

    # CLI Cache Geometry Validation
    if args.capacity_kb is not None and args.associativity is not None:
        validate_cache_geometry(args.capacity_kb, args.associativity, args.word_width_bits)

    if "all" in args.targets or args.targets == ["all"]:
        args.metrics = get_active_targets(args.tech)
    else:
        args.metrics = args.targets

    base_dir, destiny_logs_dir, plots_dir = setup_opt_dirs("opt_obj", args.run_id)

    print(f"Output dir    : {base_dir}")
    print(f"Plots dir     : {plots_dir}")
    print(f"DESTINY files : {destiny_logs_dir}")

    data_csv = os.path.join("pareto", args.tech, f"{args.tech}_pareto.csv")

    # Silently drop any metrics that are structurally zero for this technology.
    skip = set(TECH_SKIP_TARGETS.get(args.tech, []))
    rejected = [m for m in args.metrics if m in skip]
    if rejected:
        print(f"[WARN] Dropping structurally-zero metrics for {args.tech}: {rejected}")
        args.metrics = [m for m in args.metrics if m not in skip]
    if not args.metrics:
        sys.exit(f"ERROR: All requested metrics are structurally zero for {args.tech}.")

    print(f"\n{'='*80}\n  Runner Sweeper: {args.tech} | Node={args.node}nm | Roadmap={args.roadmap} | Method=gumbel (Objectives)\n  Objectives: {args.metrics}\n{'='*80}\n")

    ppa_data_frame = _load_and_filter_data(args, data_csv)
    
    # Select all Pareto-optimal baseline designs along the frontier for this technology slice
    costs = ppa_data_frame[args.metrics].values.astype(np.float64)
    pf_mask = pareto_frontier_nd(costs)
    target_vectors = ppa_data_frame[pf_mask]

    print(f"\n[2/5] Initializing InverseOptimizer (Gumbel - Objectives)")
    sizing_optimizer = InverseOptimizerGumbel(args.tech)

    print(f"\n[3/5] Solving layouts across {len(target_vectors)} target PPA metrics...")
    ppa_results = _run_optimization_sweep(target_vectors, ppa_data_frame, sizing_optimizer, args)

    ppa_results = _validate_top_designs(ppa_results, min(args.validate_top, len(ppa_results)), args, destiny_logs_dir)

    csv_path = os.path.join(destiny_logs_dir, f"benchmark_pareto_{args.tech}_{args.node}nm_{args.roadmap}_gumbel.csv")
    ppa_results[[c for c in ppa_results.columns if not c.startswith("_")]].to_csv(csv_path, index=False)
    print(f"[5/5] Sized results successfully logged -> {csv_path}")

    _plot_results_side_effect(ppa_results, ppa_data_frame, target_vectors, args, plots_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Runs one configuration + produces one csv + plot"""

import os, sys, argparse, subprocess, warnings, time, threading, shutil, re, glob, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from inverse_design import InverseOptimizer
from destiny_utils import (
    pareto_frontier_2d,
    pareto_step_line,
    cap_colormap,
    format_log_axis,
    add_cap_colorbar,
    derive_sram_physical_params,
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
    """Orchestrates layout generation, simulator execution, and physical parameter extraction.
    
    Four-tier fallback strategy:
      Tier 1: Force everything (bank + mat)
      Tier 2: Free mat, keep bank forced
      Tier 3: Free bank, keep mat forced
      Tier 4: Free both bank and mat
    """
    cell_file, cfg_file, csv_file = f"{prefix}.cell", f"{prefix}.cfg", f"{prefix}.csv"
    log_file = f"{prefix}.log"

    def _attempt(free_mat, free_bank):
        """Run DESTINY once; returns parsed metrics dict or None on failure."""
        _generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp, wn, wp, wac,
                                  read_voltage, cell_aspect_ratio, node, roadmap, opt_target, layout_config,
                                  free_mat=free_mat, free_bank=free_bank)
        process, stdout_text = _run_destiny_process(cfg_file, timeout, verbose)
        with open(log_file, "w") as f:
            f.write(stdout_text)
        csv_exists     = os.path.exists(csv_file)
        success_signal = "Finished!" in stdout_text or "csv generated successfully" in stdout_text or csv_exists
        if (process.returncode != 0 and process.returncode is not None) or "1e+50" in stdout_text or not success_signal:
            return None
        return _parse_destiny_output(stdout_text, csv_file)

    def _archive(suffix):
        """Copy current cell/cfg/log to a suffixed backup before overwriting."""
        for ext in (".cell", ".cfg", ".log"):
            src = f"{prefix}{ext}"
            if os.path.exists(src):
                shutil.copy(src, f"{prefix}_{suffix}{ext}")

    # Tier 1: force everything (bank + mat)
    result = _attempt(free_mat=False, free_bank=False)
    if result is not None:
        return result

    # Tier 2: free mat, keep bank forced
    print(f"  [warn] DESTINY tier-1 (forced bank+mat) failed; retrying with free mat (tier 2).")
    _archive("tier1")
    result = _attempt(free_mat=True, free_bank=False)
    if result is not None:
        result["_used_fallback_mat"] = True
        return result

    # Tier 3: free bank, keep mat forced
    print(f"  [warn] DESTINY tier-2 (free mat) failed; retrying with free bank (tier 3).")
    _archive("tier2")
    result = _attempt(free_mat=False, free_bank=True)
    if result is not None:
        result["_used_fallback_bank"] = True
        return result

    # Tier 4: free both bank and mat
    print(f"  [warn] DESTINY tier-3 (free bank) failed; retrying with free bank+mat (tier 4).")
    _archive("tier3")
    result = _attempt(free_mat=True, free_bank=True)
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
    """Calculates the percentage relative error between optimized surrogate predictions and target objectives."""
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
    # cache_hit_latency_ns < 100 ns, cache_area_mm2 < 1000 mm², cache_write_energy_nJ < 1000 nJ, static leakage < 10000 Watts.
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
    # Positivity filter only applies to metrics that are physically non-zero for this tech.
    active = set(get_active_targets(args.tech))
    for m in args.metrics:
        if m in active:
            ppa_df = ppa_df[ppa_df[m] > 0]
    print(f"   Library size after screening: {len(ppa_df):,}")
    if len(ppa_df) == 0: sys.exit("ERROR: Filter returned 0 matching target rows.")
    return ppa_df

def _run_optimization_sweep(target_vectors, ppa_data_frame, sizing_optimizer, args):
    """Solves sizing variables generically across any subset of the 8 dynamic target constraints."""
    from train_model import CATEGORICAL_COLS
    ppa_results_records = []
    for idx, (i, row) in enumerate(target_vectors.iterrows()):
        target_vector = {k: float(row[k]) for k in args.metrics if k in METRIC_TO_PPA_LABEL}
        ctx = row_to_context(row, args.roadmap)
        opt_ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}
        opt_ctx["data_stacked_die_count"] = 1.0 # Force 2D planar SRAM configurations
        
        # Populate one-hot encoded wire, buffer, and cell categoricals from the target row
        for col in sizing_optimizer.feature_cols:
            for cat in CATEGORICAL_COLS:
                if cat not in ["process_node_nm", "device_roadmap", "mem_cell_type"] and col.startswith(f"{cat}_"):
                    val = str(row.get(cat, ""))
                    opt_ctx[col] = 1.0 if col == f"{cat}_{val}" else 0.0
        
        snapped_design_params, surrogate_ppa_predictions, pre_snap, snapped_surrogate_ppa = sizing_optimizer.optimize(target_vector, opt_ctx, steps=args.opt_steps, verbose=args.verbose_opt)
        
        rec = {
            "is_original": False, "target_idx": i, "node_nm": int(row["process_node_nm"]), "device_roadmap": args.roadmap, "method": args.method,
            "orig_capacity_kb": row.get("capacity_kb"), "orig_word_width_bits": row.get("word_width_bits"), "orig_associativity": row.get("associativity"), "orig_data_stacked_die_count": row.get("data_stacked_die_count"),
            **{f"orig_{k}": row.get(k) for k in BENCHING_LAYOUT_COLS}, "orig_wn": ctx["_wn"], "orig_wp": ctx["_wp"], "orig_wac": ctx["_wac"], "orig_rv": ctx["_read_voltage"], "orig_ar": ctx["_cell_aspect_ratio"],
            "capacity_kb": snapped_design_params.get("capacity_kb", row.get("capacity_kb")), "word_width_bits": snapped_design_params.get("word_width_bits", row.get("word_width_bits")), "associativity": snapped_design_params.get("associativity", row.get("associativity")), "data_stacked_die_count": snapped_design_params.get("data_stacked_die_count", row.get("data_stacked_die_count")),
            **{k: snapped_design_params.get(k, row.get(k)) for k in BENCHING_LAYOUT_COLS},
            "opt_wn": snapped_design_params.get("CellInput_SRAMCellNMOSWidth (F)", ctx["_wn"]), "opt_wp": snapped_design_params.get("CellInput_SRAMCellPMOSWidth (F)", ctx["_wp"]), "opt_wac": snapped_design_params.get("CellInput_AccessCMOSWidth (F)", ctx["_wac"]), "opt_rv": snapped_design_params.get("CellInput_ReadVoltage (V)", ctx["_read_voltage"]), "opt_ar": snapped_design_params.get("CellInput_CellAspectRatio", ctx["_cell_aspect_ratio"]),
            "_wn": snapped_design_params.get("CellInput_SRAMCellNMOSWidth (F)", ctx["_wn"]), "_wp": snapped_design_params.get("CellInput_SRAMCellPMOSWidth (F)", ctx["_wp"]), "_wac": snapped_design_params.get("CellInput_AccessCMOSWidth (F)", ctx["_wac"]), "_read_voltage": snapped_design_params.get("CellInput_ReadVoltage (V)", ctx["_read_voltage"]), "_cell_aspect_ratio": snapped_design_params.get("CellInput_CellAspectRatio", ctx["_cell_aspect_ratio"]),
            "_temp": SRAM_TEMPERATURE_MAP.get(snapped_design_params.get("data_stacked_die_count", 1), ctx["_temp"]),
            "variant_name": row.get("variant_name"),
        }
        
        surr_errs, snap_errs = [], []
        for k in args.metrics:
            target_val = float(row[k])
            rec[f"target_{k}"] = target_val
            label = METRIC_TO_PPA_LABEL.get(k, "")
            
            s_val = surrogate_ppa_predictions.get(label) if surrogate_ppa_predictions else None
            rec[f"surr_{k}"] = s_val
            s_err = pct_err(s_val, target_val)
            rec[f"surr_err_{k}_pct"] = s_err
            if not np.isnan(s_err): surr_errs.append(abs(s_err))
            
            snap_val = snapped_surrogate_ppa.get(label) if snapped_surrogate_ppa else None
            rec[f"post_snap_surr_{k}"] = snap_val
            snap_err = pct_err(snap_val, target_val)
            rec[f"post_snap_surr_err_{k}_pct"] = snap_err
            if not np.isnan(snap_err): snap_errs.append(abs(snap_err))
 
        rec["surr_mean_abs_err_pct"] = float(np.mean(surr_errs)) if surr_errs else float("nan")
        rec["post_snap_surr_mean_abs_err_pct"] = float(np.mean(snap_errs)) if snap_errs else float("nan")
        
        for k in args.metrics:
            rec[f"destiny_{k}"] = None
            rec[f"destiny_err_{k}_pct"] = None
            rec[f"orig_destiny_{k}"] = None
            rec[f"orig_destiny_err_{k}_pct"] = None
        rec["destiny_mean_abs_err_pct"] = None
        rec["destiny_used_fallback_mat"] = None
        rec["destiny_used_fallback_bank"] = None
        rec["destiny_used_fallback"] = None
        rec["orig_destiny_mean_abs_err_pct"] = None
        rec["orig_destiny_used_fallback_mat"] = None
        rec["orig_destiny_used_fallback_bank"] = None
        rec["orig_destiny_used_fallback"] = None
 
        for k, v in pre_snap.items(): rec[f"pre_snap_{k}"] = v
        ppa_results_records.append(rec)
        
        print(f"  [{idx+1:3d}/{len(target_vectors)}] Solved Target Point #{idx} Sizing Design:\n"
              f"         Mean Pre-snap Surrogate |Err|:  {rec['surr_mean_abs_err_pct']:.2f}%\n"
              f"         Mean Post-snap Surrogate |Err|: {rec['post_snap_surr_mean_abs_err_pct']:.2f}%")
              
    return pd.DataFrame(ppa_results_records)

def _validate_top_designs(ppa_results, n_validate, args, work_dir):
    """Executes physical verification loop for top layout solutions."""
    if n_validate <= 0:
        print("\n[4/5] Skipping physical validator DESTINY validation runs (--validate-top 0)\n")
        return ppa_results

    print(f"\n[4/5] Running physical validation for top-{n_validate} optimal configurations...\n")
    for rank, ri in enumerate(ppa_results["post_snap_surr_mean_abs_err_pct"].sort_values().head(n_validate).index, 1):
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

            dest_errs = []
            for k in args.metrics:
                dx = destiny_ppa.get(k)
                ppa_results.at[ri, f"destiny_{k}"] = dx
                if dx is not None:
                    err = pct_err(dx, row[f"target_{k}"])
                    ppa_results.at[ri, f"destiny_err_{k}_pct"] = err
                    dest_errs.append(abs(err))
                    
            d_mean = np.mean(dest_errs) if dest_errs else float("nan")
            ppa_results.at[ri, "destiny_mean_abs_err_pct"] = d_mean
            print(f"    Validation Solver PPA Extracted -> Mean Physical Residual: {d_mean:.2f}%\n")
        else:
            print("    [warn] Validation failed.\n")

        # 2. Run validation on the original configuration
        orig_layout_config = _layout_from_row(row, prefix="orig_")
        orig_prefix = os.path.join(work_dir, f"rank{rank}_orig")
        
        orig_temp = int(SRAM_TEMPERATURE_MAP.get(max(1, int(row.orig_data_stacked_die_count)), 350))
        
        orig_destiny_ppa = validate_and_capture(
            tech=args.tech, cap_kb=int(row.orig_capacity_kb), ww=int(row.orig_word_width_bits),
            assoc=int(row.orig_associativity), stack=max(1, int(row.orig_data_stacked_die_count)),
            temp=orig_temp, wn=float(row.orig_wn), wp=float(row.orig_wp), wac=float(row.orig_wac), read_voltage=float(row.orig_rv), cell_aspect_ratio=float(row.orig_ar),
            node=int(row.get("node_nm", args.node)), roadmap=args.roadmap, timeout=args.destiny_timeout, verbose=args.verbose_destiny, opt_target="ReadLatency",
            layout_config=orig_layout_config, prefix=orig_prefix
        )

        if orig_destiny_ppa is not None:
            orig_fb_mat = orig_destiny_ppa.get("_used_fallback_mat", False)
            orig_fb_bank = orig_destiny_ppa.get("_used_fallback_bank", False)
            ppa_results.at[ri, "orig_destiny_used_fallback_mat"] = orig_fb_mat
            ppa_results.at[ri, "orig_destiny_used_fallback_bank"] = orig_fb_bank
            ppa_results.at[ri, "orig_destiny_used_fallback"] = orig_fb_mat or orig_fb_bank

            orig_dest_errs = []
            for k in args.metrics:
                dx = orig_destiny_ppa.get(k)
                ppa_results.at[ri, f"orig_destiny_{k}"] = dx
                if dx is not None:
                    err = pct_err(dx, row[f"target_{k}"])
                    ppa_results.at[ri, f"orig_destiny_err_{k}_pct"] = err
                    orig_dest_errs.append(abs(err))
                    
            orig_d_mean = np.mean(orig_dest_errs) if orig_dest_errs else float("nan")
            ppa_results.at[ri, "orig_destiny_mean_abs_err_pct"] = orig_d_mean
            print(f"    Validation Solver PPA (Original) Extracted -> Mean Physical Residual: {orig_d_mean:.2f}%\n")
        else:
            print("    [warn] Validation on original config failed.\n")
            
    return ppa_results

# Analysis & Ranking

def _compute_and_report_trend_rank(args):
    """Aggregates all method sweep run records to rank optimization algorithms by their mean absolute error percentage across all active hardware targets."""
    out_dir = Path(args.output_dir)
    csv_files = glob.glob(str(out_dir / f"benchmark_pareto_{args.tech}_{args.node}nm_{args.roadmap}_*.csv"))
    if not csv_files: return
        
    all_dfs = []
    for f in csv_files:
        try:
            df_run = pd.read_csv(f)
            if not df_run.empty: all_dfs.append(df_run)
        except Exception:
            continue
    if not all_dfs: return
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    metric_col = "destiny_mean_abs_err_pct" if "destiny_mean_abs_err_pct" in df_all.columns and df_all["destiny_mean_abs_err_pct"].notna().any() else "post_snap_surr_mean_abs_err_pct"
    df_target_means = df_all.groupby(["method", "target_idx"])[metric_col].mean().reset_index()
    
    performance_ranking_dataframe = df_target_means.groupby("method")[metric_col].agg(["mean", "median", "count"]).reset_index()
    
    # Calculate rank correlations per method
    k0 = args.metrics[0]
    target_col = f"target_{k0}"
    
    corr_records = []
    for method in df_all["method"].unique():
        df_method = df_all[df_all["method"] == method]
        pred_col = f"destiny_{k0}" if (f"destiny_{k0}" in df_method.columns and df_method[f"destiny_{k0}"].notna().any()) else f"post_snap_surr_{k0}"
        
        if target_col in df_method.columns and pred_col in df_method.columns:
            df_valid = df_method.dropna(subset=[target_col, pred_col])
            if len(df_valid) >= 3:
                df_valid = df_valid.sort_values(by=target_col)
                tau = df_valid[target_col].corr(df_valid[pred_col], method="kendall")
                rho = df_valid[target_col].corr(df_valid[pred_col], method="spearman")
            else:
                tau, rho = float("nan"), float("nan")
        else:
            tau, rho = float("nan"), float("nan")
            
        corr_records.append({
            "method": method,
            "kendall_tau": tau,
            "spearman_rho": rho
        })
    df_corrs = pd.DataFrame(corr_records)
    
    performance_ranking_dataframe = pd.merge(performance_ranking_dataframe, df_corrs, on="method", how="left")
    
    performance_ranking_dataframe = performance_ranking_dataframe.sort_values(by="mean", ascending=True)
    performance_ranking_dataframe["trend_rank"] = range(1, len(performance_ranking_dataframe) + 1)
    
    # Format floats nicely for print
    print_df = performance_ranking_dataframe.copy()
    print_df["kendall_tau"] = print_df["kendall_tau"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    print_df["spearman_rho"] = print_df["spearman_rho"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    print("\n" + "=" * 80)
    print(f"  Method Performance Trend Ranking (Target Index Grouped over {args.node}nm | {args.roadmap} | Target: {k0})")
    print("=" * 80)
    sub_df = print_df[["trend_rank", "method", "mean", "median", "count", "kendall_tau", "spearman_rho"]]
    try:
        print(sub_df.to_markdown(index=False))
    except ImportError:
        print(sub_df.to_string(index=False))
    print("=" * 80 + "\n")

def _plot_results_side_effect(ppa_results, ppa_data_frame, target_vectors, args, plots_dir):
    """Two-panel plot:
    Left  — 2D scatter of first two metrics against the library Pareto background.
    Right — per-metric bar chart showing surrogate vs DESTINY % error relative to
            the target for every active metric, one bar group per result row.
    """
    if len(args.metrics) < 2:
        return

    has_destiny = (
        "destiny_mean_abs_err_pct" in ppa_results.columns
        and ppa_results["destiny_mean_abs_err_pct"].notna().any()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)

    # ── Left: 2D scatter (first two metrics) ─────────────────────────────────
    ax = axes[0]
    x_col, y_col = args.metrics[0], args.metrics[1]
    norm, _ = cap_colormap(ppa_data_frame["capacity_kb"])

    ax.scatter(
        ppa_data_frame[x_col], ppa_data_frame[y_col],
        c=ppa_data_frame["capacity_kb"], norm=norm, cmap=plt.cm.viridis,
        s=12, alpha=0.1, linewidths=0, zorder=1, label="Library Background"
    )
    ax.scatter(
        target_vectors[x_col], target_vectors[y_col],
        facecolors="none", s=45, marker="o", edgecolors="k",
        linewidths=1.0, zorder=2, label="PPA Targets"
    )

    mask_snap = (
        ppa_results[f"post_snap_surr_{x_col}"].notna()
        & ppa_results[f"post_snap_surr_{y_col}"].notna()
    )
    if mask_snap.any():
        ax.scatter(
            ppa_results.loc[mask_snap, f"post_snap_surr_{x_col}"],
            ppa_results.loc[mask_snap, f"post_snap_surr_{y_col}"],
            c="#ffa657", s=50, marker="x", zorder=3, label="Snapped Surrogate"
        )

    mask_dest = (
        ppa_results[f"destiny_{x_col}"].notna()
        & ppa_results[f"destiny_{y_col}"].notna()
    )
    if mask_dest.any():
        ax.scatter(
            ppa_results.loc[mask_dest, f"destiny_{x_col}"],
            ppa_results.loc[mask_dest, f"destiny_{y_col}"],
            c="#3fb950", s=70, marker="*", edgecolors="k",
            linewidths=0.5, zorder=4, label="DESTINY Physical"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(METRIC_META[x_col]["label"])
    ax.set_ylabel(METRIC_META[y_col]["label"])
    format_log_axis(ax, axis="both")
    ax.set_title("Inverse Design vs Pareto Points", fontsize=10, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    add_cap_colorbar(fig, [ax], norm)

    # ── Right: per-metric % error bar chart ───────────────────────────────────
    # For each result row that has a target, show surrogate error and (if available)
    # DESTINY error as grouped bars. One group per metric, one bar per result row.
    ax2 = axes[1]

    n_metrics = len(args.metrics)
    n_rows = len(ppa_results)
    group_width = 0.7
    bar_width = group_width / (2 if has_destiny else 1) / n_rows
    x_pos = np.arange(n_metrics)

    colors_surr = plt.cm.Blues(np.linspace(0.4, 0.85, n_rows))
    colors_dest = plt.cm.Greens(np.linspace(0.4, 0.85, n_rows))

    for row_idx, (_, row) in enumerate(ppa_results.iterrows()):
        surr_errs = []
        dest_errs = []
        for m in args.metrics:
            target_val = row.get(f"target_{m}", np.nan)
            surr_val   = row.get(f"post_snap_surr_{m}", np.nan)
            dest_val   = row.get(f"destiny_{m}", np.nan)

            surr_errs.append(
                pct_err(surr_val, target_val)
                if pd.notna(target_val) and pd.notna(surr_val) else np.nan
            )
            dest_errs.append(
                pct_err(dest_val, target_val)
                if has_destiny and pd.notna(target_val) and pd.notna(dest_val) else np.nan
            )

        # Surrogate bars
        offset_surr = (row_idx - n_rows / 2 + 0.25) * bar_width * (2 if has_destiny else 1)
        bars = ax2.bar(
            x_pos + offset_surr, surr_errs,
            width=bar_width, color=colors_surr[row_idx],
            label=f"Surrogate row {row_idx}" if row_idx == 0 else "_nolegend_",
            alpha=0.85, edgecolor="none"
        )

        # DESTINY bars
        if has_destiny:
            offset_dest = offset_surr + bar_width
            ax2.bar(
                x_pos + offset_dest, dest_errs,
                width=bar_width, color=colors_dest[row_idx],
                label=f"DESTINY row {row_idx}" if row_idx == 0 else "_nolegend_",
                alpha=0.85, edgecolor="none"
            )

    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [METRIC_META[m]["label"] if m in METRIC_META else m for m in args.metrics],
        rotation=30, ha="right", fontsize=8
    )
    ax2.set_ylabel("% Error vs Target")
    ax2.set_title("Per-Metric Surrogate vs DESTINY Error", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)

    # Manual legend for surrogate vs destiny distinction
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4c9be8", label="Snapped Surrogate")]
    if has_destiny:
        legend_elements.append(Patch(facecolor="#3fb950", label="DESTINY Physical"))
    ax2.legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.suptitle(
        f"{args.tech} | {args.node}nm | {args.roadmap} | method={args.method}",
        fontsize=11, fontweight="bold"
    )
    fig.savefig(
        os.path.join(
            plots_dir,
            f"benchmark_trajectory_{args.tech}_{args.node}nm_{args.roadmap}_{args.method}.png"
        ),
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Multi-metric cache inverse sizing benchmark engine.")
    p.add_argument("--tech", default="SRAM")
    p.add_argument("--node", type=int, default=32, help="Process node nm constraint")
    p.add_argument("--roadmap", default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--method", default="baseline", choices=["baseline", "ste", "gumbel"], help="Optimizer execution engine")
    p.add_argument("--metrics", nargs="+", default=None, choices=list(METRIC_META), help="PPA objectives to size for (default: all active metrics for --tech)")
    p.add_argument("--opt-steps", type=int, default=120, help="Sizing optimization gradient steps")
    p.add_argument("--validate-top", type=int, default=5, help="Number of solved layouts to validate with C++ compiler")
    p.add_argument("--destiny-timeout", type=int, default=600, help="TIMEOUT for physical compilation subprocesses")
    p.add_argument("--output-dir", default="benchmark_results")
    p.add_argument("--max-targets", type=int, default=5, help="Maximum index constraints to evaluate from library")
    p.add_argument("--capacity-kb", type=float, default=None, help="Filter the target vectors to a specific capacity (in KB)")
    p.add_argument("--verbose-destiny", action="store_true", help="Print compiler outputs directly to console")
    p.add_argument("--verbose-opt", action="store_true", help="Print tabular design updates during steps")
    args = p.parse_args()
    if args.metrics is None:
        args.metrics = get_active_targets(args.tech)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(args.output_dir, "destiny_files", timestamp)
    plots_dir = os.path.join(args.output_dir, "validation_plots", timestamp)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Output dir    : {args.output_dir}")
    print(f"Plots dir     : {plots_dir}")
    print(f"DESTINY files : {work_dir}")

    data_csv = os.path.join("pareto", args.tech, f"{args.tech}_pareto.csv")

    # Silently drop any metrics that are structurally zero for this technology.
    skip = set(TECH_SKIP_TARGETS.get(args.tech, []))
    rejected = [m for m in args.metrics if m in skip]
    if rejected:
        print(f"[WARN] Dropping structurally-zero metrics for {args.tech}: {rejected}")
        args.metrics = [m for m in args.metrics if m not in skip]
    if not args.metrics:
        sys.exit(f"ERROR: All requested metrics are structurally zero for {args.tech}.")

    print(f"\n{'='*80}\n  Runner Sweeper: {args.tech} | Node={args.node}nm | Roadmap={args.roadmap} | Method={args.method}\n  Targets: {args.metrics}\n{'='*80}\n")

    ppa_data_frame = _load_and_filter_data(args, data_csv)
    target_vectors = ppa_data_frame.head(args.max_targets)

    print(f"\n[2/5] Initializing InverseOptimizer (method={args.method})")
    if args.method == "baseline": sizing_optimizer = InverseOptimizer(args.tech)
    elif args.method == "ste":
        from inverse_design_ste import InverseOptimizerSTE
        sizing_optimizer = InverseOptimizerSTE(args.tech)
    elif args.method == "gumbel":
        from inverse_design_gumbel import InverseOptimizerGumbel
        sizing_optimizer = InverseOptimizerGumbel(args.tech)
    else:
        sys.exit(f"ERROR: Sizing variant method {args.method} is unsupported.")

    print(f"\n[3/5] Solving layouts across {len(target_vectors)} target PPA metrics...")
    ppa_results = _run_optimization_sweep(target_vectors, ppa_data_frame, sizing_optimizer, args)

    ppa_results = _validate_top_designs(ppa_results, min(args.validate_top, len(ppa_results)), args, work_dir)

    csv_path = os.path.join(args.output_dir, f"benchmark_pareto_{args.tech}_{args.node}nm_{args.roadmap}_{args.method}.csv")
    ppa_results[[c for c in ppa_results.columns if not c.startswith("_")]].to_csv(csv_path, index=False)
    print(f"[5/5] Sized results successfully logged -> {csv_path}")

    _compute_and_report_trend_rank(args)
    _plot_results_side_effect(ppa_results, ppa_data_frame, target_vectors, args, plots_dir)

if __name__ == "__main__":
    main()
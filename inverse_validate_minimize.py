#!/usr/bin/env python3
"""
validate_minimize.py — Physical validation for inverse_design_gumbel_minimize.py output.

Reads the CSV produced by::

    python inverse_design_gumbel_minimize.py ...   # appends to runs/results.csv by default

and for every row:
  1. Reconstructs the DESTINY .cell + .cfg files from the snapped design
     parameters stored in the CSV.
  2. Runs the DESTINY simulator binary (./destiny) and parses its stdout.
  3. Prints a side-by-side table: surrogate (post-snap) vs physical truth.
  4. Generates a Pareto plot comparing the evaluated point to the Pareto frontier.

Falls back to free-mat subarray selection if the forced-mat attempt fails.

Output layout (all paths derived from --runs-dir, default: runs/):
    runs/
        results.csv          ← produced by inverse_design_gumbel_minimize.py
        validated.csv        ← written by this script
        validation_plots/    ← one PNG per row
        destiny_files/
            <timestamp>/     ← .cell, .cfg, .log for every DESTINY invocation

Usage
-----
    python validate_minimize.py                        # uses runs/results.csv
    python validate_minimize.py --runs-dir my_runs
    python validate_minimize.py --input-csv path/to/other.csv
    python validate_minimize.py --timeout 120 --verbose
"""

import argparse
import ast
import datetime
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from destiny_utils import (
    TARGET_COLS as TARGET_KEYS,
    TARGET_SHORT_LABELS,
    METRIC_META,
    derive_sram_physical_params,
    get_active_targets,
    cap_colormap,
    format_log_axis,
)

_LAYOUT_COLS = [
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SRAM_TEMPERATURE_MAP: Dict[int, int] = {1: 300, 2: 363, 4: 380}

# Mapping: TARGET_KEYS key → SHORT_LABEL (e.g. "cache_hit_latency_ns" → "ReadLatency")
_KEY_TO_SHORT: Dict[str, str] = dict(zip(TARGET_KEYS, TARGET_SHORT_LABELS))


# ---------------------------------------------------------------------------
# DESTINY invocation helpers  (adapted from inverse_design_sweep.py)
# ---------------------------------------------------------------------------

def _map_repeater(val: str) -> str:
    v = str(val)
    if "Fully-Optimized" in v or "Opt" in v:
        return "RepeatedOpt"
    if "No" in v:
        return "RepeatedNone"
    return v.replace(" ", "")


def _write_cell_file(path: str, wn: float, wp: float, wac: float,
                     read_voltage: float, cell_aspect_ratio: float,
                     node: int) -> None:
    """Write a DESTINY .cell file for an SRAM design."""
    cell_params = {
        "SRAMCellNMOSWidth (F)": wn,
        "SRAMCellPMOSWidth (F)": wp,
        "AccessCMOSWidth (F)": wac,
    }
    derive_sram_physical_params(cell_params, node)
    content = (
        f"-MemCellType: SRAM\n"
        f"-CellArea (F^2): {cell_params['CellArea (F^2)']:.5f}\n"
        f"-SRAMCellNMOSWidth (F): {wn:.4f}\n"
        f"-SRAMCellPMOSWidth (F): {wp:.4f}\n"
        f"-AccessCMOSWidth (F): {wac:.4f}\n"
        f"-AccessType: CMOS\n"
        f"-MinSenseVoltage (mV): {cell_params['MinSenseVoltage (mV)']:.4f}\n"
        f"-CellAspectRatio: {cell_aspect_ratio:.4f}\n"
        f"-ReadVoltage (V): {read_voltage:.4f}\n"
        f"-Stitching: 16\n"
    )
    with open(path, "w") as f:
        f.write(content)


def _write_cfg_file(path: str, cell_path: str,
                    cap_kb: int, ww: int, assoc: int, stack: int, temp: int,
                    node: int, roadmap: str,
                    layout: Dict, free_mat: bool = False, free_bank: bool = False) -> None:
    """Write a DESTINY .cfg file, injecting all Force* directives from *layout*."""
    cfg = (
        f"-OptimizationTarget: ReadLatency\n"
        f"-EnablePruning: Yes\n"
        f"-Capacity (KB): {cap_kb}\n"
        f"-WordWidth (bit): {ww}\n"
        f"-Associativity (for cache only): {assoc}\n"
        f"-StackedDieCount: {stack}\n"
        f"-Temperature (K): {temp}\n"
        f"-DeviceRoadmap: {roadmap}\n"
        f"-MemoryCellInputFile: {os.path.abspath(cell_path)}\n"
        f"-ProcessNode: {node}\n"
    )

    def _iv(key):
        v = layout.get(key)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

    if _iv("data_mux_sense_amp") is not None:
        cfg += f"-ForceMuxSenseAmp: {int(float(_iv('data_mux_sense_amp')))}\n"
    if _iv("data_mux_output_lev1") is not None:
        cfg += f"-ForceMuxOutputLev1: {int(float(_iv('data_mux_output_lev1')))}\n"
    if _iv("data_mux_output_lev2") is not None:
        cfg += f"-ForceMuxOutputLev2: {int(float(_iv('data_mux_output_lev2')))}\n"

    if not free_bank:
        amc = _iv("data_num_active_mat_per_col")
        amr = _iv("data_num_active_mat_per_row")
        if amc is not None and amr is not None:
            c, r = int(float(amc)), int(float(amr))
            cfg += f"-ForceBank (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"

    if not free_mat:
        asc = _iv("data_num_active_subarray_per_col")
        asr = _iv("data_num_active_subarray_per_row")
        if asc is not None and asr is not None:
            c, r = int(float(asc)), int(float(asr))
            cfg += f"-ForceMat (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"

    # Tag array
    if _iv("tag_area_optimization_level") is not None:
        val = str(_iv("tag_area_optimization_level")).lower()
        val = "latency" if "latency" in val else ("area" if "area" in val else val)
        cfg += f"-TagBufferDesignOptimization: {val}\n"
    if _iv("tag_local_wire_type") is not None:
        cfg += f"-TagLocalWireType: {str(_iv('tag_local_wire_type')).replace(' ', '')}\n"
    if _iv("tag_local_wire_repeater_type") is not None:
        cfg += f"-TagLocalWireRepeaterType: {_map_repeater(_iv('tag_local_wire_repeater_type'))}\n"
    if _iv("tag_local_wire_low_swing") is not None:
        cfg += f"-TagLocalWireUseLowSwing: {_iv('tag_local_wire_low_swing')}\n"
    if _iv("tag_global_wire_type") is not None:
        cfg += f"-TagGlobalWireType: {str(_iv('tag_global_wire_type')).replace(' ', '')}\n"
    if _iv("tag_global_wire_repeater_type") is not None:
        cfg += f"-TagGlobalWireRepeaterType: {_map_repeater(_iv('tag_global_wire_repeater_type'))}\n"
    if _iv("tag_global_wire_low_swing") is not None:
        cfg += f"-TagGlobalWireUseLowSwing: {_iv('tag_global_wire_low_swing')}\n"

    if _iv("tag_mux_output_lev1") is not None:
        cfg += f"-ForceTagMuxOutputLev1: {int(float(_iv('tag_mux_output_lev1')))}\n"
    if _iv("tag_mux_output_lev2") is not None:
        cfg += f"-ForceTagMuxOutputLev2: {int(float(_iv('tag_mux_output_lev2')))}\n"
    if _iv("tag_mux_sense_amp") is not None:
        cfg += f"-ForceTagMuxSenseAmp: {int(float(_iv('tag_mux_sense_amp')))}\n"

    if not free_bank:
        tnrm = _iv("tag_num_row_mat")
        tncm = _iv("tag_num_col_mat")
        tanr = _iv("tag_num_active_mat_per_row")
        tanc = _iv("tag_num_active_mat_per_col")
        if tnrm is not None and tncm is not None:
            tr, tc = int(float(tnrm)), int(float(tncm))
            ar = int(float(tanr)) if tanr is not None else tr
            ac = int(float(tanc)) if tanc is not None else tc
            cfg += f"-ForceTagBank (Total AxB, Active CxD): {tr}x{tc}, {ac}x{ar}\n"
        elif tanr is not None and tanc is not None:
            r, c = int(float(tanr)), int(float(tanc))
            cfg += f"-ForceTagBank (Total AxB, Active CxD): {r}x{c}, {r}x{c}\n"

    if not free_mat:
        tnrs = _iv("tag_num_row_subarray")
        tncs = _iv("tag_num_col_subarray")
        tnas_r = _iv("tag_num_active_subarray_per_row")
        tnas_c = _iv("tag_num_active_subarray_per_col")
        if tnrs is not None and tncs is not None:
            tr, tc = int(float(tnrs)), int(float(tncs))
            ar = int(float(tnas_r)) if tnas_r is not None else tr
            ac = int(float(tnas_c)) if tnas_c is not None else tc
            cfg += f"-ForceTagMat (Total AxB, Active CxD): {tr}x{tc}, {ac}x{ar}\n"
        elif tnas_r is not None and tnas_c is not None:
            r, c = int(float(tnas_r)), int(float(tnas_c))
            cfg += f"-ForceTagMat (Total AxB, Active CxD): {r}x{c}, {r}x{c}\n"

    # Data array wiring
    if _iv("data_local_wire_type") is not None:
        cfg += f"-LocalWireType: {str(_iv('data_local_wire_type')).replace(' ', '')}\n"
    if _iv("data_local_wire_repeater_type") is not None:
        cfg += f"-LocalWireRepeaterType: {_map_repeater(_iv('data_local_wire_repeater_type'))}\n"
    if _iv("data_local_wire_low_swing") is not None:
        cfg += f"-LocalWireUseLowSwing: {_iv('data_local_wire_low_swing')}\n"
    if _iv("data_global_wire_type") is not None:
        cfg += f"-GlobalWireType: {str(_iv('data_global_wire_type')).replace(' ', '')}\n"
    if _iv("data_global_wire_repeater_type") is not None:
        cfg += f"-GlobalWireRepeaterType: {_map_repeater(_iv('data_global_wire_repeater_type'))}\n"
    if _iv("data_global_wire_low_swing") is not None:
        cfg += f"-GlobalWireUseLowSwing: {_iv('data_global_wire_low_swing')}\n"
    if _iv("data_area_optimization_level") is not None:
        val = str(_iv("data_area_optimization_level")).lower()
        val = "latency" if "latency" in val else ("area" if "area" in val else val)
        cfg += f"-BufferDesignOptimization: {val}\n"

    with open(path, "w") as f:
        f.write(cfg)


def _run_destiny(cfg_path: str, timeout: int, verbose: bool):
    """Spawn ./destiny, stream stdout, return (process, captured_text)."""
    lines: List[str] = []

    def _reader(pipe):
        for line in iter(pipe.readline, ""):
            lines.append(line.rstrip())
            if verbose:
                print(f"    [destiny] {line}", end="")

    proc = subprocess.Popen(
        ["./destiny", cfg_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    t = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
    t.start()
    t.join(timeout=timeout)
    if proc.poll() is None:
        proc.kill()
        print(f"  [warn] DESTINY timed out after {timeout}s for {cfg_path}")
    return proc, "\n".join(lines)


def _parse_destiny_stdout(stdout: str, csv_path: str) -> Optional[Dict[str, float]]:
    """Extract cache-level PPA from DESTINY stdout; falls back to its output CSV."""
    _LATENCY = lambda v, u: v / 1000 if u == "ps" else (v * 1000 if u == "us" else v)
    _ENERGY  = lambda v, u: v / 1000 if u == "p"  else (v * 1000 if u == "u"  else v)
    regexes = {
        "cache_area_mm2":           (r"(?:Total|Cache) Area\s*=\s*([\d.]+)\s*(mm\^2|um\^2)",
                                     lambda v, u: v / 1e6 if u == "um^2" else v),
        "cache_hit_latency_ns":     (r"(?:Read|Cache Hit) Latency\s*=\s*([\d.]+)\s*(ps|ns|us)",   _LATENCY),
        "cache_write_latency_ns":   (r"(?:Write|Cache Write) Latency\s*=\s*([\d.]+)\s*(ps|ns|us)", _LATENCY),
        "cache_refresh_latency_ns": (r"(?:Refresh|Cache Refresh) Latency\s*=\s*([\d.]+)\s*(ps|ns|us)", _LATENCY),
        "cache_hit_energy_nJ":      (r"(?:Read|Cache Hit) Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J",      _ENERGY),
        "cache_write_energy_nJ":    (r"(?:Write|Cache Write) Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J",    _ENERGY),
        "cache_refresh_energy_nJ":  (r"(?:Refresh|Cache Refresh) Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J", _ENERGY),
        "cache_leakage_mW": (
            r"(?:Leakage|Cache Total Leakage) Power\s*=\s*([\d.]+)\s*(p|n|u|m)?W",
            lambda v, u: v / 1000 if u == "u" else (v / 1e6 if u == "n" else (v / 1e9 if u == "p" else (v * 1000 if not u else v))),
        ),
    }
    result: Dict[str, float] = {}
    for key, (pat, scale) in regexes.items():
        m = re.search(pat, stdout)
        if m:
            result[key] = scale(float(m.group(1)), m.group(2) if m.lastindex >= 2 else "")

    if not result and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        result = {c: float(df[c].iloc[0]) for c in regexes if c in df.columns}

    return result or None


def _validate_row(
    row: pd.Series,
    prefix: str,
    timeout: int,
    verbose: bool,
) -> Optional[Dict[str, float]]:
    """
    Write cell+cfg, run DESTINY, return parsed physical PPA dict (or None).
    Attempts forced-mat first; on failure retries with free mat selection.
    Both attempt files are preserved with '_forced' suffix on fallback.
    """
    cell_path = f"{prefix}.cell"
    cfg_path  = f"{prefix}.cfg"
    log_path  = f"{prefix}.log"
    csv_path  = cfg_path.replace(".cfg", ".csv")

    # Extract cell parameters from the row
    wn  = float(row.get("CellInput_SRAMCellNMOSWidth (F)",
                         row.get("opt_wn", 2.2)))
    wp  = float(row.get("CellInput_SRAMCellPMOSWidth (F)",
                         row.get("opt_wp", 1.0)))
    wac = float(row.get("CellInput_AccessCMOSWidth (F)",
                         row.get("opt_wac", 1.1)))
    rv  = float(row.get("CellInput_ReadVoltage (V)",
                         row.get("opt_rv", 1.0)))
    ar  = float(row.get("CellInput_CellAspectRatio",
                         row.get("opt_ar", 1.0)))

    # Reconstruct process node from one-hot columns (process_node_nm_32, etc.)
    node = int(row.get("node_nm", 32))
    roadmap = str(row.get("roadmap", "HP"))
    cap_kb  = int(row.get("capacity_kb", 2))
    ww      = int(row.get("word_width_bits", 128))
    assoc   = int(row.get("associativity", 4))
    stack   = max(1, int(float(row.get("data_stacked_die_count", 1))))
    temp    = int(row.get("temperature_K",
                           SRAM_TEMPERATURE_MAP.get(stack, 300)))

    layout = {k: row.get(k) for k in _LAYOUT_COLS}

    def _attempt(free_mat: bool, free_bank: bool) -> Optional[Dict[str, float]]:
        _write_cell_file(cell_path, wn, wp, wac, rv, ar, node)
        _write_cfg_file(cfg_path, cell_path,
                        cap_kb, ww, assoc, stack, temp,
                        node, roadmap, layout, free_mat=free_mat, free_bank=free_bank)
        proc, stdout = _run_destiny(cfg_path, timeout, verbose)
        with open(log_path, "w") as f:
            f.write(stdout)

        csv_exists = os.path.exists(csv_path)
        success    = "Finished!" in stdout or "csv generated successfully" in stdout or csv_exists
        failed     = proc.returncode not in (0, None) or "1e+50" in stdout or not success
        return None if failed else _parse_destiny_stdout(stdout, csv_path)

    def _archive(suffix: str):
        for ext in (".cell", ".cfg", ".log"):
            src = f"{prefix}{ext}"
            if os.path.exists(src):
                shutil.copy(src, f"{prefix}_{suffix}{ext}")

    # Tier 1: force everything (bank + mat)
    result = _attempt(free_mat=False, free_bank=False)
    if result is not None:
        return result

    # Tier 2: free mat, keep bank forced
    print("  [warn] DESTINY tier-1 (forced bank+mat) failed; retrying with free mat (tier 2).")
    _archive("tier1")
    result = _attempt(free_mat=True, free_bank=False)
    if result is not None:
        result["_used_fallback_mat"] = True
        return result

    # Tier 3: free bank, keep mat forced
    print("  [warn] DESTINY tier-2 (free mat) failed; retrying with free bank (tier 3).")
    _archive("tier2")
    result = _attempt(free_mat=False, free_bank=True)
    if result is not None:
        result["_used_fallback_bank"] = True
        return result

    # Tier 4: free both bank and mat
    print("  [warn] DESTINY tier-3 (free bank) failed; retrying with free bank+mat (tier 4).")
    _archive("tier3")
    result = _attempt(free_mat=True, free_bank=True)
    if result is not None:
        result["_used_fallback_mat"] = True
        result["_used_fallback_bank"] = True
    return result

# ---------------------------------------------------------------------------
# Surrogate value extraction from the CSV row
# ---------------------------------------------------------------------------

def _surrogate_ppa(row: pd.Series, active_keys: List[str]) -> Dict[str, float]:
    """
    Return {metric_key: surrogate_value} from the post-snap predictions
    stored in the optimizer CSV row.
    Tries "post_snap_pred_{SHORT_LABEL}" first, then "pred_{SHORT_LABEL}".
    """
    out: Dict[str, float] = {}
    for k in active_keys:
        short = _KEY_TO_SHORT.get(k, "")
        val = row.get(f"post_snap_pred_{k}")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = row.get(f"post_snap_pred_{short}")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = row.get(f"pred_{k}")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = row.get(f"pred_{short}")
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            out[k] = float(val)
    return out


# ---------------------------------------------------------------------------
# Plotting logic
# ---------------------------------------------------------------------------
def _plot_validation(
    row_idx: int,
    row: pd.Series,
    surr: Dict[str, float],
    phys: Dict[str, float],
    objectives: List[str],
    tech: str,
    node: int,
    roadmap: str,
    plots_dir: str,
):
    """Generates a Pareto plot with the validation points, saved to *plots_dir*."""
    cap_kb = int(row.get("capacity_kb", 0))
    if len(objectives) < 2:
        print("  [plot] Not enough objectives to plot a 2D Pareto frontier.")
        return
        
    x_col, y_col = objectives[0], objectives[1]
    
    pareto_csv = os.path.join("pareto", tech, f"{tech}_pareto.csv")
    df_lib = pd.read_csv(pareto_csv)
    df_lib = df_lib[
        (df_lib["mem_cell_type"].str.upper() == tech.upper()) &
        (df_lib["process_node_nm"] == node) &
        (df_lib["device_roadmap"].str.upper() == roadmap.upper())
    ]
    
    if len(df_lib) == 0:
        print(f"  [plot] No Pareto points found for {tech} {node}nm {roadmap} {cap_kb}KB")
        return
        
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    norm, _ = cap_colormap(df_lib["capacity_kb"])
    
    # Plot library background
    ax.scatter(df_lib[x_col], df_lib[y_col], c=df_lib["capacity_kb"], norm=norm, cmap=plt.cm.viridis, s=20, alpha=0.3, linewidths=0, zorder=1, label="Library Background")
    
    # Plot surrogate and physical
    if surr.get(x_col) is not None and surr.get(y_col) is not None:
        ax.scatter(surr[x_col], surr[y_col], c="#ffa657", s=100, marker="x", zorder=3, label="Surrogate Prediction")
    else:
        print(f"  [plot] Missing surrogate values for {x_col} or {y_col}. surr keys: {list(surr.keys())}")
        
    if phys.get(x_col) is not None and phys.get(y_col) is not None:
        ax.scatter(phys[x_col], phys[y_col], c="#3fb950", s=120, marker="*", edgecolors="k", linewidths=0.5, zorder=4, label="Physical Validated")
    else:
        print(f"  [plot] Missing physical values for {x_col} or {y_col}. phys keys: {list(phys.keys())}")
        
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    xlabel = METRIC_META.get(x_col, {}).get("label", x_col)
    ylabel = METRIC_META.get(y_col, {}).get("label", y_col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_log_axis(ax, axis="both")
    
    ax.set_title(f"Validation for Minimize Output (Row {row_idx})\n{tech} | {node}nm | {roadmap} | {cap_kb}KB", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    
    out_img = os.path.join(plots_dir, f"validation_plot_row{row_idx:04d}.png")
    fig.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved Pareto plot to {out_img}")

# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def _print_row_report(
    row_idx: int,
    row: pd.Series,
    surr: Dict[str, float],
    phys: Dict[str, float],
    objectives: List[str],
    active_keys: List[str],
    used_fallback: bool,
    fallback_type: str = "",
) -> Dict[str, float]:
    """
    Print the surrogate vs physical comparison table for one design row.
    Returns a dict of {metric_key: pct_error} for building the output CSV.
    """
    node     = int(row.get("node_nm", "?"))
    roadmap  = str(row.get("roadmap", "?"))
    tech     = str(row.get("tech", "SRAM"))
    cap_kb   = row.get("capacity_kb", "?")
    ww       = row.get("word_width_bits", "?")

    fallback_tag = f"  [fallback {fallback_type}]" if used_fallback else ""
    print(f"\n{'─'*70}")
    print(f"  Row {row_idx}  |  {tech}  {node}nm {roadmap}  "
          f"{cap_kb}KB / {ww}b{fallback_tag}")
    print(f"  Objectives: {objectives}")
    print(f"{'─'*70}")

    w_met  = max((len(METRIC_META.get(k, {}).get("label", k)) for k in active_keys), default=20) + 2
    w_unit = 6
    w_val  = 14

    header = (
        f"  {'Metric':<{w_met}}"
        f"  {'Unit':>{w_unit}}"
        f"  {'Surrogate':>{w_val}}"
        f"  {'Physical':>{w_val}}"
        f"  {'Err %':>8}"
        f"  {'Obj?':>5}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    pct_errors: Dict[str, float] = {}
    for k in active_keys:
        label = METRIC_META.get(k, {}).get("label", k)
        unit  = METRIC_META.get(k, {}).get("unit", "")
        s_val = surr.get(k)
        p_val = phys.get(k)

        if s_val is not None and p_val is not None and p_val != 0:
            err = (p_val - s_val) / abs(s_val) * 100.0
        else:
            err = float("nan")
        pct_errors[k] = err

        s_str = f"{s_val:.4g}" if s_val is not None else "—"
        p_str = f"{p_val:.4g}" if p_val is not None else "—"
        e_str = f"{err:+.1f}%" if not math.isnan(err) else "  —"
        is_obj = "★" if k in objectives else ""

        print(
            f"  {label:<{w_met}}"
            f"  {unit:>{w_unit}}"
            f"  {s_str:>{w_val}}"
            f"  {p_str:>{w_val}}"
            f"  {e_str:>8}"
            f"  {is_obj:>5}"
        )

    valid_errs = [abs(v) for v in pct_errors.values() if not math.isnan(v)]
    if valid_errs:
        print(f"\n  Mean |err|: {np.mean(valid_errs):.2f}%   "
              f"(objectives only: "
              f"{np.mean([abs(pct_errors[k]) for k in objectives if not math.isnan(pct_errors.get(k, float('nan')))]) :.2f}%)")

    return pct_errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Physical validation for inverse_design_gumbel_minimize.py CSV output."
    )
    p.add_argument(
        "--runs-dir", default="runs",
        help="Root directory for all run artifacts (default: runs/). "
             "Output layout: <runs-dir>/results.csv (input), validated.csv, "
             "validation_plots/, destiny_files/<timestamp>/."
    )
    p.add_argument(
        "--input-csv", default=None,
        help="Override input CSV path (default: <runs-dir>/results.csv)."
    )
    p.add_argument(
        "--output-csv", default=None,
        help="Override output validated CSV path (default: <runs-dir>/validated.csv)."
    )
    p.add_argument("--timeout", type=int, default=120,
                   help="DESTINY timeout per run in seconds (default: 120)")
    p.add_argument(
        "--work-dir", default=None,
        help="Override directory for DESTINY scratch files "
             "(default: <runs-dir>/destiny_files/<timestamp>/)."
    )
    p.add_argument("--verbose", action="store_true",
                   help="Stream DESTINY stdout to console")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Validate only the first N rows")
    args = p.parse_args()

    # ── Resolve paths from --runs-dir ─────────────────────────────────────
    runs_dir   = os.path.abspath(args.runs_dir)
    input_csv  = args.input_csv  or os.path.join(runs_dir, "results.csv")
    output_csv = args.output_csv or os.path.join(runs_dir, "validated.csv")
    plots_dir  = os.path.join(runs_dir, "validation_plots")
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir   = args.work_dir   or os.path.join(runs_dir, "destiny_files", timestamp)

    os.makedirs(runs_dir,  exist_ok=True)
    os.makedirs(work_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Runs dir      : {runs_dir}")
    print(f"Input CSV     : {input_csv}")
    print(f"Output CSV    : {output_csv}")
    print(f"Plots dir     : {plots_dir}")
    print(f"DESTINY files : {work_dir}")

    # ── Load input CSV ────────────────────────────────────────────────────
    df = pd.read_csv(input_csv)
    if df.empty:
        sys.exit("ERROR: Input CSV is empty.")

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    print(f"Loaded {len(df)} row(s) from {input_csv}")

    # ── Accumulate results ────────────────────────────────────────────────
    records = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        tech = str(row.get("tech", "SRAM")).upper()
        active_keys = get_active_targets(tech)

        # Parse stored objectives (may be a Python list literal string)
        raw_obj = row.get("objectives", "[]")
        if isinstance(raw_obj, str):
            try:
                objectives: List[str] = ast.literal_eval(raw_obj)
            except (ValueError, SyntaxError):
                objectives = [s.strip() for s in raw_obj.strip("[]").split(",") if s.strip()]
        else:
            objectives = list(raw_obj) if raw_obj else []

        # Surrogate predictions already in the CSV
        surr = _surrogate_ppa(row, active_keys)

        # Physical validation
        prefix = os.path.join(work_dir, f"row{row_idx:04d}")
        print(f"\n[row {row_idx}] Running DESTINY ...", end=" ", flush=True)
        phys_result = _validate_row(row, prefix=prefix,
                                    timeout=args.timeout, verbose=args.verbose)

        if phys_result is None:
            print("FAILED")
            pct_errors: Dict[str, float] = {}
            fb_mat = fb_bank = used_fallback = False
            phys_result = {}
        else:
            fb_mat = bool(phys_result.pop("_used_fallback_mat", False))
            fb_bank = bool(phys_result.pop("_used_fallback_bank", False))
            used_fallback = fb_mat or fb_bank
            fallback_type = {(True, True): "both", (True, False): "mat", (False, True): "bank"}.get(
                (fb_mat, fb_bank), ""
            )
            print("OK" + (f" (fallback {fallback_type})" if used_fallback else ""))
            pct_errors = _print_row_report(
                row_idx, row, surr, phys_result,
                objectives, active_keys, used_fallback, fallback_type=fallback_type
            )

        node     = int(row.get("node_nm", 32))
        roadmap  = str(row.get("roadmap", "HP"))
        _plot_validation(row_idx, row, surr, phys_result, objectives, tech, node, roadmap,
                         plots_dir=plots_dir)

        # Build output record
        rec = dict(row)
        rec["destiny_used_fallback_mat"]  = fb_mat
        rec["destiny_used_fallback_bank"] = fb_bank
        rec["destiny_used_fallback"]      = used_fallback
        for k in active_keys:
            rec[f"destiny_{k}"]               = phys_result.get(k) if phys_result else None
            rec[f"surr_{k}"]                  = surr.get(k)
            rec[f"destiny_vs_surr_err_{k}_pct"] = pct_errors.get(k, float("nan"))

        valid_errs = [abs(pct_errors[k]) for k in active_keys
                      if not math.isnan(pct_errors.get(k, float("nan")))]
        rec["destiny_mean_abs_err_pct"] = float(np.mean(valid_errs)) if valid_errs else float("nan")

        obj_errs = [abs(pct_errors.get(k, float("nan"))) for k in objectives
                    if not math.isnan(pct_errors.get(k, float("nan")))]
        rec["destiny_obj_mean_abs_err_pct"] = float(np.mean(obj_errs)) if obj_errs else float("nan")

        records.append(rec)

    # ── Summary ───────────────────────────────────────────────────────────
    df_out = pd.DataFrame(records)

    valid_runs = df_out["destiny_mean_abs_err_pct"].notna()
    n_valid    = int(valid_runs.sum())
    n_total    = len(df_out)

    print(f"\n{'='*70}")
    print(f"  Summary: {n_valid}/{n_total} rows validated successfully")
    if n_valid > 0:
        mean_all = df_out.loc[valid_runs, "destiny_mean_abs_err_pct"].mean()
        mean_obj = df_out.loc[valid_runs, "destiny_obj_mean_abs_err_pct"].mean()
        print(f"  Mean |err| across all active metrics : {mean_all:.2f}%")
        print(f"  Mean |err| across optimised objectives: {mean_obj:.2f}%")
    print(f"{'='*70}\n")

    # ── Write output CSV ──────────────────────────────────────────────────
    out_cols = [c for c in df_out.columns if not c.startswith("_")]
    df_out[out_cols].to_csv(output_csv, index=False)
    print(f"Validated results written to {output_csv}")


if __name__ == "__main__":
    main()

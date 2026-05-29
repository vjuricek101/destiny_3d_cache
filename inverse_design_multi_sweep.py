#!/usr/bin/env python3
"""
Combinatorial sweep across optimizer variants x feasibility x metric pairs.
Saves one CSV per configuration; use plot_multi_sweep.py to plot
----------------
  variant    : ["baseline", "ste", "gumbel"]
  feasibility: [False, True]
  metric_pair: all combinations of the four (x_metric, y_metric) pairs built
               from _BASE_PAIRS (both orderings of each base pair)

Each CSV is written to --output-dir with the filename:
  sweep_{tech}_{node}nm_{roadmap}_{variant}_{x_metric}_vs_{y_metric}_feas{0|1}.csv
"""

import os
import sys
import argparse
import itertools
import warnings
import time
import threading
import shutil
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
warnings.filterwarnings("ignore", category=UserWarning)

import torch
torch.set_num_threads(1)

from destiny_utils import (
    METRIC_META,
    METRIC_TO_PPA_LABEL,
    LAYOUT_COLS as _LAYOUT_COLS
)

# Temperature co-varies with stack count (from run_exploration.py).
SRAM_TEMPERATURE_MAP = {1: 300, 2: 363, 4: 380}

# -- DESTINY validation helpers (mirrored from inverse_design_sweep.py) --------

def _generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp,
                               wn, wp, wac, read_voltage, node, roadmap, opt_target, layout):
    """Generate .cell and .cfg files for a DESTINY validation run."""
    from destiny_utils import derive_sram_physical_params
    p = {"SRAMCellNMOSWidth (F)": wn, "SRAMCellPMOSWidth (F)": wp, "AccessCMOSWidth (F)": wac}
    derive_sram_physical_params(p, node)

    cell_content = f"""-MemCellType: SRAM
-CellArea (F^2): {p["CellArea (F^2)"]:.4f}
-SRAMCellNMOSWidth (F): {wn:.4f}
-SRAMCellPMOSWidth (F): {wp:.4f}
-AccessCMOSWidth (F): {wac:.4f}
-AccessType: CMOS
-MinSenseVoltage (mV): {p["MinSenseVoltage (mV)"]:.4f}
-CellAspectRatio: 1.4600
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
    # Append forced layout parameters
    if layout.get("mux_sa")    is not None: cfg_content += f"-ForceMuxSenseAmp: {int(layout['mux_sa'])}\n"
    if layout.get("mux_ol2")   is not None: cfg_content += f"-ForceMuxOutputLev2: {int(layout['mux_ol2'])}\n"
    if layout.get("act_mat_col") is not None and layout.get("act_mat_row") is not None:
        c, r = int(layout["act_mat_col"]), int(layout["act_mat_row"])
        cfg_content += f"-ForceBank (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"
    if layout.get("act_sub_col") is not None and layout.get("act_sub_row") is not None:
        c, r = int(layout["act_sub_col"]), int(layout["act_sub_row"])
        cfg_content += f"-ForceMat (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"

    with open(cell_file, "w") as f: f.write(cell_content)
    with open(cfg_file,  "w") as f: f.write(cfg_content)
    return cell_content, cfg_content


def _run_destiny_process(cfg_file, timeout, verbose):
    """Execute the DESTINY subprocess and stream its stdout."""
    output_lines = []
    def _stream(pipe):
        for line in iter(pipe.readline, ""):
            output_lines.append(line.strip())
            if verbose:
                print(f"    [destiny] {line}", end="")

    process = subprocess.Popen(
        ["./destiny", cfg_file],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1,
    )
    t = threading.Thread(target=_stream, args=(process.stdout,), daemon=True)
    t.start()
    t.join(timeout=timeout)
    if process.poll() is None:
        process.kill()
        print(f"  [warn] DESTINY timed out after {timeout}s.")
    return process, "\n".join(output_lines)


def _parse_destiny_output(stdout_text, csv_file):
    """Extract PPA metrics from DESTINY stdout; fall back to CSV if needed."""
    res = {}
    m = re.search(r"Read Latency\s*=\s*([\d.]+)\s*(ps|ns|us)", stdout_text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        res["cache_hit_latency_ns"] = val / 1000 if unit == "ps" else (val * 1000 if unit == "us" else val)
    m = re.search(r"(?:Total|Cache) Area\s*=\s*([\d.]+)\s*(mm\^2|um\^2)", stdout_text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        res["cache_area_mm2"] = val / 1e6 if unit == "um^2" else val
    m = re.search(r"Write Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J", stdout_text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        res["cache_write_energy_nJ"] = val / 1000 if unit == "p" else (val * 1000 if unit == "u" else val)
    m = re.search(r"Leakage Power\s*=\s*([\d.]+)\s*(p|n|u|m)?W", stdout_text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        if unit == "u":    val /= 1000
        elif unit == "n":  val /= 1e6
        elif unit == "p":  val /= 1e9
        elif not unit:     val *= 1000   # W → mW
        res["cache_leakage_mW"] = val
    if not res and os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        res = {col: float(df[col].iloc[0]) for col in [
            "cache_hit_latency_ns", "cache_area_mm2",
            "cache_write_energy_nJ", "cache_leakage_mW",
        ]}
    return res if res else None


def _validate_and_capture(cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage,
                           node, roadmap, opt_target, layout,
                           timeout, verbose, run_tag):
    """Write configs, run DESTINY, parse results.  Returns PPA dict or None."""
    cell_file = f"multisweep_val_{run_tag}.cell"
    cfg_file  = f"multisweep_val_{run_tag}.cfg"
    csv_file  = f"multisweep_val_{run_tag}.csv"
    try:
        cell_c, cfg_c = _generate_destiny_configs(
            cell_file, cfg_file, cap_kb, ww, assoc, stack, temp,
            wn, wp, wac, read_voltage, node, roadmap, opt_target, layout,
        )
        if verbose:
            print(f"\n  [debug] CFG:\n{cfg_c}\n  [debug] CELL:\n{cell_c}")
        process, stdout_text = _run_destiny_process(cfg_file, timeout, verbose)
        csv_exists     = os.path.exists(csv_file)
        success_signal = ("Finished!" in stdout_text
                          or "csv generated successfully" in stdout_text
                          or csv_exists)
        if ((process.returncode != 0 and process.returncode is not None)
                or "1e+50" in stdout_text
                or not success_signal):
            if not verbose:
                rc = process.returncode if process.returncode is not None else "timeout"
                print(f"    x DESTINY failed (exit {rc}).")
            os.makedirs("debug_configs", exist_ok=True)
            ts = int(time.time() * 1000)
            for src in [cfg_file, cell_file]:
                if os.path.exists(src):
                    shutil.copy(src, f"debug_configs/FAILED_{ts}_{os.path.basename(src)}")
            return None
        return _parse_destiny_output(stdout_text, csv_file)
    except Exception as e:
        print(f"  [error] Subprocess error: {e}")
        return None
    finally:
        for f in [cell_file, cfg_file, csv_file]:
            if os.path.exists(f):
                os.remove(f)


def _layout_from_row(row):
    """Extract forced layout params from a result row; returns dict with keys
    mux_sa, mux_ol2, act_mat_col/row, act_sub_col/row."""
    mapping = {
        "data_mux_sense_amp":              "mux_sa",
        "data_mux_output_lev2":            "mux_ol2",
        "data_num_active_mat_per_col":     "act_mat_col",
        "data_num_active_mat_per_row":     "act_mat_row",
        "data_num_active_subarray_per_col": "act_sub_col",
        "data_num_active_subarray_per_row": "act_sub_row",
    }
    out = {}
    for src, dst in mapping.items():
        val = row.get(src) if isinstance(row, dict) else (
            row[src] if src in row.index else None
        )
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[dst] = int(val)
    return out


def _validate_top_designs(df_res, metrics, n_validate,
                           roadmap, node, destiny_timeout, verbose_destiny,
                           run_tag_prefix):
    """Run DESTINY on the *n_validate* designs with the lowest surrogate error.
    """
    if n_validate <= 0:
        return df_res

    # Initialise DESTINY result columns if they don't exist yet
    for k in metrics:
        if f"destiny_{k}" not in df_res.columns:
            df_res[f"destiny_{k}"] = None
        if f"destiny_err_{k}_pct" not in df_res.columns:
            df_res[f"destiny_err_{k}_pct"] = None
    if "destiny_mean_abs_err_pct" not in df_res.columns:
        df_res["destiny_mean_abs_err_pct"] = None

    print(f"   DESTINY validation for top-{n_validate} designs "
          f"(ranked by post-snap surrogate error)...")

    ranked_idx = df_res["post_snap_surr_mean_abs_err_pct"].sort_values().head(n_validate).index
    for rank, ri in enumerate(ranked_idx, 1):
        row = df_res.loc[ri]
        opt_target = "ReadLatency"   # conservative default; collapses wire loops

        # Recover private cell params stored by run_single
        wn  = float(row.get("opt_wn",  row.get("orig_wn",  2.5)))
        wp  = float(row.get("opt_wp",  row.get("orig_wp",  2.0)))
        wac = float(row.get("opt_wac", row.get("orig_wac", 2.5)))
        rv  = float(row.get("opt_rv",  row.get("orig_rv",  1.0)))
        stack_raw = row.get("data_stacked_die_count", 0)
        stack = max(1, int(stack_raw)) if stack_raw is not None else 1
        temp  = SRAM_TEMPERATURE_MAP.get(stack, 300)

        node_val = int(row.get("node_nm", node))
        cap_kb   = int(row["capacity_kb"])
        ww       = int(row["word_width_bits"])
        assoc    = int(row["associativity"])

        layout   = _layout_from_row(row)
        run_tag  = f"{run_tag_prefix}_{rank}"

        print(f"   [{rank}/{n_validate}] Validating "
              f"Cap={cap_kb}KB  WW={ww}  Assoc={assoc}  Stack={stack}  "
              f"Node={node_val}nm  OptTarget={opt_target}  "
              f"post_snap_err={row['post_snap_surr_mean_abs_err_pct']:.1f}%")

        destiny_ppa = _validate_and_capture(
            cap_kb=cap_kb, ww=ww, assoc=assoc, stack=stack, temp=temp,
            wn=wn, wp=wp, wac=wac, read_voltage=rv,
            node=node_val, roadmap=roadmap, opt_target=opt_target,
            layout=layout,
            timeout=destiny_timeout, verbose=verbose_destiny,
            run_tag=run_tag,
        )

        if destiny_ppa is None:
            print("     x DESTINY failed.")
            continue

        d_errs = []
        out_strs = []
        for k in metrics:
            dk = destiny_ppa.get(k)
            df_res.at[ri, f"destiny_{k}"] = dk
            if dk is not None:
                err = pct_err(dk, row[f"target_{k}"])
                df_res.at[ri, f"destiny_err_{k}_pct"] = err
                d_errs.append(abs(err))
                out_strs.append(f"{k}={dk:.4g} (err={err:+.1f}%)")
            else:
                df_res.at[ri, f"destiny_err_{k}_pct"] = None
        
        d_mean = float(np.mean(d_errs)) if d_errs else float("nan")
        df_res.at[ri, "destiny_mean_abs_err_pct"] = d_mean

        print(f"     DESTINY → " + "  ".join(out_strs) + f"  mean={d_mean:.1f}%")

    return df_res


def _validate_trend_designs(df_res, x_col, y_col, n_trend,
                            roadmap, node, destiny_timeout, verbose_destiny,
                            run_tag_prefix, arch_params):
    """Run DESTINY on *n_trend* designs uniformly spaced along the target sequence."""
    if n_trend <= 0 or len(df_res) == 0:
        return df_res

    # Initialise DESTINY trend columns
    for col in ["trend_destiny_x", "trend_destiny_y"]:
        if col not in df_res.columns:
            df_res[col] = None

    print(f"   DESTINY validation for {n_trend} trend points...")

    # Sort by target_x to get a monotonic sequence
    df_sorted = df_res.sort_values("target_x").reset_index(drop=False)
    
    # Pick n_trend evenly spaced indices
    indices = np.linspace(0, len(df_sorted) - 1, min(n_trend, len(df_sorted)), dtype=int)
    trend_df = df_sorted.iloc[indices]
    
    for rank, (ri, row_tuple) in enumerate(zip(trend_df["index"], trend_df.itertuples()), 1):
        # We need the row as a Series for _layout_from_row and easier access
        row = df_res.loc[ri]
        
        opt_target = "ReadLatency"   # conservative default

        wn  = float(row.get("opt_wn",  row.get("orig_wn",  2.5)))
        wp  = float(row.get("opt_wp",  row.get("orig_wp",  2.0)))
        wac = float(row.get("opt_wac", row.get("orig_wac", 2.5)))
        rv  = float(row.get("opt_rv",  row.get("orig_rv",  1.0)))
        stack_raw = row.get("data_stacked_die_count", 0)
        stack = max(1, int(stack_raw)) if stack_raw is not None else 1
        temp  = SRAM_TEMPERATURE_MAP.get(stack, 300)

        node_val = int(row.get("node_nm", node))
        cap_kb   = int(row["capacity_kb"])
        ww       = int(row["word_width_bits"])
        assoc    = int(row["associativity"])

        layout   = _layout_from_row(row)
        run_tag  = f"{run_tag_prefix}_trend_{rank}"

        print(f"   [{rank}/{n_trend}] Trend Validation "
              f"Cap={cap_kb}KB  WW={ww}  Assoc={assoc}  Stack={stack}  "
              f"Node={node_val}nm  TargetX={row['target_x']:.4g}")

        destiny_ppa = _validate_and_capture(
            cap_kb=cap_kb, ww=ww, assoc=assoc, stack=stack, temp=temp,
            wn=wn, wp=wp, wac=wac, read_voltage=rv,
            node=node_val, roadmap=roadmap, opt_target=opt_target,
            layout=layout,
            timeout=destiny_timeout, verbose=verbose_destiny,
            run_tag=run_tag,
        )

        if destiny_ppa is None:
            print("     x DESTINY failed.")
            continue

        df_res.at[ri, "trend_destiny_x"] = destiny_ppa.get(x_col)
        df_res.at[ri, "trend_destiny_y"] = destiny_ppa.get(y_col)
        
        print(f"     DESTINY → x={destiny_ppa.get(x_col):.4g}  y={destiny_ppa.get(y_col):.4g}")

    # Compute correlation
    valid_trend = df_res.dropna(subset=["trend_destiny_x", "trend_destiny_y"])
    if len(valid_trend) > 1:
        tau_x, _ = kendalltau(valid_trend["target_x"], valid_trend["trend_destiny_x"])
        tau_y, _ = kendalltau(valid_trend["target_y"], valid_trend["trend_destiny_y"])
        spr_x, _ = spearmanr(valid_trend["target_x"], valid_trend["trend_destiny_x"])
        spr_y, _ = spearmanr(valid_trend["target_y"], valid_trend["trend_destiny_y"])
        
        print(f"   Trend Metrics (n={len(valid_trend)}):")
        print(f"     X-axis ({x_col}): Kendall Tau = {tau_x:.3f}, Spearman = {spr_x:.3f}")
        print(f"     Y-axis ({y_col}): Kendall Tau = {tau_y:.3f}, Spearman = {spr_y:.3f}")
        
        corr_file = f"sweep_correlations_{arch_params['tech']}_{arch_params['node']}nm_{arch_params['roadmap']}.csv"
        corr_path = os.path.join(arch_params['output_dir'], corr_file)
        
        row_dict = {
            "variant": arch_params['variant'],
            "feasibility": arch_params['feasibility'],
            "x_metric": x_col,
            "y_metric": y_col,
            "n_points": len(valid_trend),
            "kendall_tau_x": tau_x,
            "spearman_x": spr_x,
            "kendall_tau_y": tau_y,
            "spearman_y": spr_y
        }
        
        file_exists = os.path.exists(corr_path)
        with open(corr_path, "a") as f:
            if not file_exists:
                f.write(",".join(row_dict.keys()) + "\n")
            f.write(",".join(str(row_dict[k]) for k in row_dict.keys()) + "\n")
            
    return df_res


# -- Sweep definition ---------------------------------------------------------

VARIANTS = ["baseline", "ste", "gumbel"]

ALL_METRICS = [
    "cache_hit_latency_ns",
    "cache_area_mm2",
    "cache_write_energy_nJ",
    "cache_leakage_mW",
]

FEASIBILITY_FLAGS = [False, True]

# -- Frozen 2-D constraint ----------------------------------------------------

FROZEN_PARAMS = {"data_stacked_die_count": 1.0}

# -- Output filename ----------------------------------------------------------

def _csv_filename(args, node, roadmap, variant, metrics, feasibility):
    """Build the canonical CSV filename for one sweep configuration."""
    feas_tag = "1" if feasibility else "0"
    metrics_str = "_vs_".join(metrics)
    return f"sweep_{args.tech}_{node}nm_{roadmap}_{variant}_{metrics_str}_feas{feas_tag}.csv"

# -- Optimizer loader ---------------------------------------------------------

def _load_optimizer(variant, tech, use_feasibility):
    """Instantiate the correct optimizer class for *variant*."""
    if variant == "baseline":
        from inverse_design import InverseOptimizer
        return InverseOptimizer(tech, use_feasibility=use_feasibility)
    elif variant == "ste":
        from inverse_design_ste import InverseOptimizer as InverseOptimizerSTE
        return InverseOptimizerSTE(tech, use_feasibility=use_feasibility)
    elif variant == "gumbel":
        from inverse_design_gumbel import InverseOptimizerGumbel
        return InverseOptimizerGumbel(tech, use_feasibility=use_feasibility)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

# -- Data helpers (reused from inverse_design_sweep.py) ----------------------

def _load_and_filter_data(args, data_csv, node, roadmap, metrics):
    """Load dataset and apply standard filters."""
    print(f"   Loading dataset: {data_csv}")
    df_all = pd.read_csv(data_csv)
    df = df_all[
        (df_all["mem_cell_type"].str.upper() == args.tech.upper()) &
        (df_all["process_node_nm"] == node) &
        (df_all["device_roadmap"].str.upper() == roadmap.upper())
    ].copy()

    df = df[
        (df["cache_hit_latency_ns"] < 100) & (df["cache_area_mm2"] < 1000) &
        (df["cache_write_energy_nJ"]  < 1000) & (df["cache_leakage_mW"] > 0) &
        (df["cache_leakage_mW"] < 1e7) 
    ]
    for m in metrics:
        df = df[df[m] > 0]
    print(f"   Rows after filter: {len(df):,}")
    if len(df) == 0:
        return None
    return df


def _select_targets(df, metrics, max_targets):
    """Return the target DataFrame (Pareto), optionally capped."""
    from destiny_utils import pareto_frontier_nd

    costs = df[list(metrics)].values
    mask = pareto_frontier_nd(costs)
    df_target = df[mask].reset_index(drop=True)
    print(f"   Pareto-optimal points: {len(df_target)}")

    if max_targets is not None:
        df_target = df_target.head(max_targets)
        print(f"   Capped to {len(df_target)} targets (--max-targets={max_targets})")
    return df_target


def row_to_context(row, roadmap):
    """Extract fixed physical context from a dataset row."""
    node = int(row["process_node_nm"])
    ctx  = {f"process_node_nm_{node}": 1.0, "temperature_K": float(row.get("temperature_K", 350.0))}
    for rm in ["HP", "LOP", "LSTP"]:
        ctx[f"device_roadmap_{rm}"] = 1.0 if rm == roadmap else 0.0
    ctx["_wn"]           = float(row.get("CellInput_SRAMCellNMOSWidth (F)", 2.5))
    ctx["_wp"]           = float(row.get("CellInput_SRAMCellPMOSWidth (F)", 2.0))
    ctx["_wac"]          = float(row.get("CellInput_AccessCMOSWidth (F)",   2.5))
    ctx["_read_voltage"] = float(row.get("CellInput_ReadVoltage (V)", 1.0))
    ctx["_temp"]         = int(row.get("temperature_K", 350))
    if hasattr(row, "index") and "opt_target" in row.index:
        ctx["_opt_target"] = str(row["opt_target"])
    return ctx


def pct_err(predicted, target):
    """Signed percentage error."""
    return float("nan") if target == 0 else (predicted - target) / abs(target) * 100.0

# -- Core single-run function -------------------------------------------------

def run_single(args, node, roadmap, variant, metrics, feasibility, csv_path):
    """
    Run one complete sweep configuration and save results to *csv_path*.

    Returns True on success, False if the data filter yields no rows or the
    target selection fails.
    """
    data_csv = os.path.join(
        "pareto", args.tech,
        f"{args.tech}_full_data.csv"
    )

    print(f"\n  {'─'*64}")
    print(f"  node={node}nm  roadmap={roadmap}  variant={variant}  feasibility={feasibility}")
    print(f"  metrics={metrics}")
    print(f"  {'─'*64}")

    # ── 1. Load & filter ──────────────────────────────────────────────────────
    df = _load_and_filter_data(args, data_csv, node, roadmap, metrics)
    if df is None:
        return False

    # ── 2. Select targets ─────────────────────────────────────────────────────
    df_target = _select_targets(df, metrics, args.max_targets)
    if df_target is None or len(df_target) == 0:
        return False

    # ── 3. Load optimizer ─────────────────────────────────────────────────────
    print(f"   Loading optimizer variant={variant!r}  feasibility={feasibility}")
    try:
        opt = _load_optimizer(variant, args.tech, feasibility)
    except Exception as e:
        print(f"   ERROR loading optimizer: {e}")
        return False

    # ── 4. Optimization loop ──────────────────────────────────────────────────
    print(f"   Running inverse optimization for {len(df_target)} targets "
          f"({args.opt_steps} steps each)...\n")

    records = []
    for i, row in df_target.iterrows():
        targets = {k: row[k] for k in metrics if k in METRIC_TO_PPA_LABEL}

        ctx     = row_to_context(row, roadmap)
        opt_ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}
        # Apply the frozen 2-D constraint before calling optimize() so the
        # optimizer treats stacked_die_count as a fixed input, not a free var.
        opt_ctx.update(FROZEN_PARAMS)

        try:
            design, ppa, pre_snap, snapped_ppa = opt.optimize(
                targets, opt_ctx,
                steps=args.opt_steps,
                verbose=args.verbose_opt,
            )
        except Exception as e:
            print(f"    [i={i}] optimize() raised: {e}")
            continue

        surr_vals = {k: ppa.get(METRIC_TO_PPA_LABEL.get(k, "")) for k in metrics}
        snap_vals = {k: snapped_ppa.get(METRIC_TO_PPA_LABEL.get(k, "")) for k in metrics} if snapped_ppa else {k: None for k in metrics}

        errs = [abs(pct_err(surr_vals[k], row[k])) for k in metrics if surr_vals[k] is not None]
        surr_mean_err = float(np.mean(errs)) if errs else float("nan")

        snap_errs = [abs(pct_err(snap_vals[k], row[k])) for k in metrics if snap_vals[k] is not None]
        snap_mean_err = float(np.mean(snap_errs)) if snap_errs else float("nan")

        opt_wn  = design.get("CellInput_SRAMCellNMOSWidth (F)", ctx["_wn"])
        opt_wp  = design.get("CellInput_SRAMCellPMOSWidth (F)", ctx["_wp"])
        opt_wac = design.get("CellInput_AccessCMOSWidth (F)",   ctx["_wac"])
        opt_rv  = design.get("CellInput_ReadVoltage (V)",        ctx["_read_voltage"])

        record = {
            "variant": variant, "feasibility": feasibility,
            "metrics_optimized": " ".join(metrics),
            "is_original": False, "target_idx": i,
            "node_nm": int(row["process_node_nm"]),
        }
        for k in metrics:
            record[f"target_{k}"] = row[k]
            record[f"target_percentile_{k}"] = float((df[k] <= row[k]).mean() * 100)
            record[f"surr_{k}"] = surr_vals[k]
            record[f"surr_err_{k}_pct"] = pct_err(surr_vals[k], row[k]) if surr_vals[k] is not None else None
            record[f"post_snap_surr_{k}"] = snap_vals[k]
            record[f"post_snap_surr_err_{k}_pct"] = pct_err(snap_vals[k], row[k]) if snap_vals[k] is not None else None

        record["surr_mean_abs_err_pct"] = surr_mean_err
        record["post_snap_surr_mean_abs_err_pct"] = snap_mean_err

        # ── original design from training data ─────────────────────────
        record.update({
            "orig_capacity_kb":            row.get("capacity_kb"),
            "orig_word_width_bits":        row.get("word_width_bits"),
            "orig_associativity":          row.get("associativity"),
            "orig_data_stacked_die_count": row.get("data_stacked_die_count"),
            **{f"orig_{k}": row.get(k) for k in _LAYOUT_COLS},
            "orig_wn":  ctx["_wn"],  "orig_wp":  ctx["_wp"],
            "orig_wac": ctx["_wac"], "orig_rv":  ctx["_read_voltage"],
        })
        # ── optimized design inputs ────────────────────────────────────
        record.update({
            "capacity_kb":           design.get("capacity_kb"),
            "word_width_bits":       design.get("word_width_bits"),
            "associativity":         design.get("associativity"),
            "data_stacked_die_count": design.get("data_stacked_die_count"),
            **{k: design.get(k) for k in _LAYOUT_COLS},
            "opt_wn": opt_wn, "opt_wp": opt_wp, "opt_wac": opt_wac, "opt_rv": opt_rv,
        })
        # ── pre-snap continuous values ─────────────────────────────────
        record.update({f"pre_snap_{k}": v for k, v in pre_snap.items()})

        records.append(record)

        if args.verbose_opt or (i % max(1, len(df_target) // 10) == 0):
            target_str = "  ".join(f"{k}={row[k]:.4g}" for k in metrics)
            surr_str = "  ".join(f"{k}={surr_vals[k]:.4g}" for k in metrics)
            print(
                f"    [{i+1:3d}/{len(df_target)}] Target: {target_str}\n"
                f"           Pre-snap Surr:  {surr_str}  (err={surr_mean_err:.1f}%)\n"
            )

    if not records:
        print("   WARNING: no records produced — skipping CSV write.")
        return False

    df_res = pd.DataFrame(records)

    # ── 5. DESTINY validation ─────────────────────────────────────────────────
    if args.validate_top > 0:
        node_val = args.nodes[0] if (args.nodes and len(args.nodes) == 1) else node
        # Build a unique tag so parallel runs don't collide on temp file names
        run_tag = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(os.path.basename(csv_path))[0])
        df_res = _validate_top_designs(
            df_res, metrics,
            n_validate=min(args.validate_top, len(df_res)),
            roadmap=roadmap,
            node=node_val,
            destiny_timeout=args.destiny_timeout,
            verbose_destiny=args.verbose_destiny,
            run_tag_prefix=run_tag,
        )

    if args.n_trend_points > 0 and len(metrics) == 2:
        x_col, y_col = metrics[0], metrics[1]
        node_val = args.nodes[0] if (args.nodes and len(args.nodes) == 1) else node
        run_tag = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(os.path.basename(csv_path))[0])
        arch_params = {
            "tech": args.tech, "node": node_val, "roadmap": roadmap,
            "variant": variant, "feasibility": feasibility, "output_dir": args.output_dir
        }
        df_res = _validate_trend_designs(
            df_res, x_col, y_col,
            n_trend=args.n_trend_points,
            roadmap=roadmap,
            node=node_val,
            destiny_timeout=args.destiny_timeout,
            verbose_destiny=args.verbose_destiny,
            run_tag_prefix=run_tag,
            arch_params=arch_params
        )

    # Strip private underscore columns before saving
    save_cols = [c for c in df_res.columns if not c.startswith("_")]
    df_res[save_cols].to_csv(csv_path, index=False)

    s_errs      = df_res["surr_mean_abs_err_pct"].dropna()
    post_s_errs = df_res["post_snap_surr_mean_abs_err_pct"].dropna()
    d_errs      = df_res["destiny_mean_abs_err_pct"].dropna() if "destiny_mean_abs_err_pct" in df_res.columns else pd.Series([], dtype=float)
    summary = (
        f"\n   ✓ Saved → {csv_path}\n"
        f"     Pre-snap  mean |err|: {s_errs.mean():.2f}%  (n={len(s_errs)})\n"
        f"     Post-snap mean |err|: {post_s_errs.mean():.2f}%  (n={len(post_s_errs)})"
    )
    if len(d_errs) > 0:
        summary += f"\n     DESTINY        mean |err|: {d_errs.mean():.2f}%  (n={len(d_errs)})"
    print(summary)
    return True

# -- Main ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Combinatorial sweep over optimizer variants × feasibility × metric pairs. "
            "Saves one CSV per configuration for later grid plotting."
        )
    )
    p.add_argument("--tech",        default="SRAM",
                   help="Memory cell technology (default: SRAM)")
    p.add_argument("--nodes",       type=int, nargs="+", default=None,
                   help="Process nodes [nm] to sweep (default: all nodes in dataset)")
    p.add_argument("--roadmaps",    nargs="+", default=None, choices=["HP", "LOP", "LSTP"],
                   help="Device roadmaps to sweep (default: all roadmaps in dataset)")
    p.add_argument("--metrics",     nargs="+", default=None,
                   help="Specific metrics to optimize (default: sweep all 15 combinations of 4 metrics)")
    p.add_argument("--opt-steps",       type=int, default=400,
                   help="Gradient steps per optimization run (default: 400)")
    p.add_argument("--output-dir",      default="multi_sweep_results",
                   help="Directory to write per-configuration CSV files")
    p.add_argument("--max-targets",     type=int, default=None,
                   help="Max target designs to optimize per run (default: all)")
    p.add_argument("--validate-top",    type=int, default=0,
                   help="DESTINY-validate the top-N lowest-surrogate-error designs "
                        "per run (default: 0 = disabled)")
    p.add_argument("--n-trend-points",  type=int, default=0,
                   help="Number of evenly spaced points along the target sequence to validate for trend verification (default: 0)")
    p.add_argument("--destiny-timeout", type=int, default=30,
                   help="Per-call DESTINY timeout in seconds (default: 30)")
    p.add_argument("--verbose-opt",     action="store_true",
                   help="Print pre/post-snap parameter table for every optimized design")
    p.add_argument("--verbose-destiny", action="store_true",
                   help="Print DESTINY cell/cfg and full stdout for every validation call")
    p.add_argument("--variants", nargs="+", default=["baseline", "ste", "gumbel"], choices=["baseline", "ste", "gumbel"],
                   help="Optimizer variants to sweep (default: all three)")
    p.add_argument("--feasibility-only", action="store_true",
                   help="Only evaluate configurations with feasibility=True enabled")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_csv = os.path.join(
        "pareto", args.tech,
        f"{args.tech}_full_data.csv"
    )

    if not os.path.exists(data_csv):
        sys.exit(f"ERROR: Dataset not found at {data_csv}")

    print(f"Loading dataset to identify nodes and roadmaps: {data_csv}")
    df_all = pd.read_csv(data_csv)
    df_tech = df_all[df_all["mem_cell_type"].str.upper() == args.tech.upper()]

    if args.nodes is None:
        nodes = sorted(df_tech["process_node_nm"].dropna().unique().tolist())
    else:
        nodes = args.nodes

    if args.roadmaps is None:
        roadmaps = sorted(df_tech["device_roadmap"].dropna().unique().tolist())
    else:
        roadmaps = args.roadmaps

    import itertools
    if args.metrics:
        combos_list = [tuple(args.metrics)]
    else:
        combos_list = []
        for r in range(1, len(ALL_METRICS) + 1):
            combos_list.extend(list(itertools.combinations(ALL_METRICS, r)))

    # Build the full combinatorial grid
    feas_flags = [True] if args.feasibility_only else FEASIBILITY_FLAGS
    combos = list(itertools.product(nodes, roadmaps, args.variants, feas_flags, combos_list))
    total  = len(combos)

    print(f"\n{'='*70}")
    print(f"  Multi-Sweep: {args.tech}")
    print(f"  Nodes      : {nodes}")
    print(f"  Roadmaps   : {roadmaps}")
    print(f"  Dimensions : {len(nodes)} nodes × {len(roadmaps)} roadmaps × {len(args.variants)} variants "
          f"× {len(feas_flags)} feasibility × {len(combos_list)} metric combinations = {total} configurations")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*70}\n")

    n_done = n_skipped = n_failed = 0

    for run_idx, (node, roadmap, variant, feasibility, metrics) in enumerate(combos, 1):
        fname    = _csv_filename(args, node, roadmap, variant, metrics, feasibility)
        csv_path = os.path.join(args.output_dir, fname)

        print(f"\n[{run_idx:2d}/{total}] {fname}")

        if os.path.exists(csv_path):
            print(f"  ↷ SKIP — CSV already exists: {csv_path}")
            n_skipped += 1
            continue

        t0      = time.time()
        success = run_single(args, node, roadmap, variant, metrics, feasibility, csv_path)
        elapsed = time.time() - t0

        if success:
            n_done += 1
            print(f"  Elapsed: {elapsed:.1f}s")
        else:
            n_failed += 1
            print(f"  ✗ Run FAILED after {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print(f"  Sweep complete: {n_done} done, {n_skipped} skipped (already existed), {n_failed} failed")
    print(f"  CSV files in: {os.path.abspath(args.output_dir)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

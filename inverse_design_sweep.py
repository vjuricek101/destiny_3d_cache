#!/usr/bin/env python3
"""Runs one configuration + produces one csv + plot"""

import os, sys, argparse, subprocess, warnings, time, threading, shutil, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    LAYOUT_COLS as _LAYOUT_COLS
)

warnings.filterwarnings("ignore", category=UserWarning)

# Temperature co-varies with stack count (from run_exploration.py).
SRAM_TEMPERATURE_MAP = {1: 300, 2: 363, 4: 380}


# -- Target-point selection ----------------------------------------------------

def select_pareto(df, x_col, y_col):
    """Return the non-dominated Pareto front for x_col x y_col."""
    result = df[pareto_frontier_2d(df[x_col].values, df[y_col].values)].reset_index(drop=True)
    print(f"   Pareto-optimal points: {len(result)}")
    return result

def select_median(df, x_col, y_col, n_bins):
    """Return one median-y representative per equal-quantile x bin."""
    work  = df.copy().reset_index(drop=True)
    x_vals = work[x_col].values
    edges  = np.unique(np.percentile(x_vals, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 2:
        raise ValueError("Too few unique x values -- reduce --n-bins.")

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask   = (x_vals >= lo) & (x_vals <= hi if hi == edges[-1] else x_vals < hi)
        bucket = work[mask]
        if len(bucket):
            rows.append(work.loc[(bucket[y_col] - bucket[y_col].median()).abs().idxmin()])

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f"   Median points selected: {len(result)} (from {n_bins} requested bins)")
    return result

# -- DESTINY validation --------------------------------------------------------

def _generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage, node, roadmap, opt_target, layout):
    """Generate .cell and .cfg files for DESTINY."""
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

    # ── 6 backprop-solved layout / mux parameters ─────────────────────────────
    if layout.get("mux_sa")     is not None: cfg_content += f"-ForceMuxSenseAmp: {int(layout['mux_sa'])}\n"
    if layout.get("mux_ol2")    is not None: cfg_content += f"-ForceMuxOutputLev2: {int(layout['mux_ol2'])}\n"
    if layout.get("act_mat_col") is not None and layout.get("act_mat_row") is not None:
        c, r = int(layout["act_mat_col"]), int(layout["act_mat_row"])
        # Total = Active: search space is already collapsed to this single point.
        cfg_content += f"-ForceBank (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"
    if layout.get("act_sub_col") is not None and layout.get("act_sub_row") is not None:
        c, r = int(layout["act_sub_col"]), int(layout["act_sub_row"])
        cfg_content += f"-ForceMat (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"

    with open(cell_file, "w") as f: f.write(cell_content)
    with open(cfg_file,  "w") as f: f.write(cfg_content)
    return cell_content, cfg_content

def _run_destiny_process(cfg_file, timeout, verbose):
    """Execute DESTINY subprocess and capture output."""
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
        print(f"  [warn] DESTINY timed out after {timeout}s.")
        
    return process, "\n".join(output_lines)

def _parse_destiny_output(stdout_text, csv_file):
    """Extract PPA metrics from DESTINY stdout or fallback to CSV."""
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
        if unit == "u": val /= 1000
        elif unit == "n": val /= 1e6
        elif unit == "p": val /= 1e9
        elif not unit: val *= 1000 # W to mW
        res["cache_leakage_mW"] = val

    if not res and os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        res = {col: float(df[col].iloc[0]) for col in [
            "cache_hit_latency_ns", "cache_area_mm2", "cache_write_energy_nJ", "cache_leakage_mW"
        ]}
    return res if res else None

def validate_and_capture(tech, cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage,
                          node=32, roadmap="HP", timeout=60, verbose=False, opt_target="Full",
                          mux_sa=None, mux_ol2=None, act_mat_col=None, act_mat_row=None,
                          act_sub_col=None, act_sub_row=None):
    """Run DESTINY with given params; return PPA dict or None."""
    cell_file, cfg_file, csv_file = "validation_temp_bench.cell", "validation_temp_bench.cfg", "validation_temp_bench.csv"
    
    layout = {"mux_sa": mux_sa, "mux_ol2": mux_ol2, "act_mat_col": act_mat_col, "act_mat_row": act_mat_row, "act_sub_col": act_sub_col, "act_sub_row": act_sub_row}
    
    try:
        cell_c, cfg_c = _generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage, node, roadmap, opt_target, layout)
        if verbose: print(f"\n  [debug] CFG:\n{cfg_c}\n  [debug] CELL:\n{cell_c}")

        process, stdout_text = _run_destiny_process(cfg_file, timeout, verbose)

        csv_exists = os.path.exists(csv_file)
        success_signal = "Finished!" in stdout_text or "csv generated successfully" in stdout_text or csv_exists
        
        if (process.returncode != 0 and process.returncode is not None) or "1e+50" in stdout_text or not success_signal:
            if not verbose: print(f"    x DESTINY failed (exit {process.returncode if process.returncode is not None else 'timeout'}).")
            os.makedirs("debug_configs", exist_ok=True)
            ts = int(time.time() * 1000)
            shutil.copy(cfg_file,  f"debug_configs/FAILED_{ts}_{cfg_file}")
            shutil.copy(cell_file, f"debug_configs/FAILED_{ts}_{cell_file}")
            return None

        res = _parse_destiny_output(stdout_text, csv_file)
        
        if verbose and res:
            print(f"  [debug] DESTINY -> lat={res.get('cache_hit_latency_ns', float('nan')):.4g}ns  "
                  f"area={res.get('cache_area_mm2', float('nan')):.4g}mm2  energy={res.get('cache_write_energy_nJ', float('nan')):.4g}nJ  "
                  f"leak={res.get('cache_leakage_mW', float('nan')):.4g}mW")
        return res
    except Exception as e:
        print(f"  [error] Subprocess error: {e}")
        return None
    finally:
        for f in [cell_file, cfg_file, csv_file]:
            if os.path.exists(f): os.remove(f)

# -- Helpers -------------------------------------------------------------------

def row_to_context(row, roadmap):
    """Extract fixed physical context from a dataset row for the optimizer."""
    node = int(row["process_node_nm"])
    ctx  = {f"process_node_nm_{node}": 1.0, "temperature_K": float(row.get("temperature_K", 350.0))}
    for rm in ["HP", "LOP", "LSTP"]:
        ctx[f"device_roadmap_{rm}"] = 1.0 if rm == roadmap else 0.0
        
    # Fallback cell params for DESTINY validation only.
    ctx["_wn"]   = float(row.get("CellInput_SRAMCellNMOSWidth (F)", 2.5))
    ctx["_wp"]   = float(row.get("CellInput_SRAMCellPMOSWidth (F)", 2.0))
    ctx["_wac"]  = float(row.get("CellInput_AccessCMOSWidth (F)",   2.5))
    ctx["_read_voltage"] = float(row.get("CellInput_ReadVoltage (V)", 1.0))
    ctx["_temp"] = int(row.get("temperature_K", 350))
    
    # Pass training-row opt_target to condition the optimizer.
    if hasattr(row, "index") and "opt_target" in row.index:
        ctx["_opt_target"] = str(row["opt_target"])
    return ctx

def pct_err(predicted, target):
    """Signed percentage error."""
    return float("nan") if target == 0 else (predicted - target) / abs(target) * 100.0

def _layout_from_row(row):
    """Map a result-row's backprop-solved layout columns to the keys expected by
    _generate_destiny_configs / validate_and_capture.
    """
    mapping = {
        "data_mux_sense_amp":               "mux_sa",
        "data_mux_output_lev2":             "mux_ol2",
        "data_num_active_mat_per_col":      "act_mat_col",
        "data_num_active_mat_per_row":      "act_mat_row",
        "data_num_active_subarray_per_col": "act_sub_col",
        "data_num_active_subarray_per_row": "act_sub_row",
    }
    out = {}
    for src, dst in mapping.items():
        val = row[src] if src in row.index else None
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[dst] = int(val)
    return out

# -- Main Helpers --------------------------------------------------------------

def _load_and_filter_data(args, data_csv, x_col, y_col):
    """Load dataset and apply base filters."""
    print(f"[1/6] Loading dataset: {data_csv}")
    df_all = pd.read_csv(data_csv)
    df = df_all[
        (df_all["mem_cell_type"].str.upper() == args.tech.upper()) &
        (df_all["process_node_nm"] == args.node) &
        (df_all["device_roadmap"].str.upper() == args.roadmap.upper())
    ].copy()
    
    df = df[
        (df["cache_hit_latency_ns"] < 100) & (df["cache_area_mm2"] < 1000) &
        (df["cache_write_energy_nJ"]  < 1000) & (df["cache_leakage_mW"] > 0) &
        (df["cache_leakage_mW"] < 1e7) & (df[x_col] > 0) & (df[y_col] > 0)
    ]
    print(f"   Rows after filter: {len(df):,}")
    if len(df) == 0: sys.exit("ERROR: No rows match the filter.")
    return df

def _run_optimization_sweep(df_target, df, opt, x_col, y_col, x_label, y_label, x_unit, y_unit, args):
    """Run inverse design for each target point."""
    records = []
    for i, row in df_target.iterrows():
        target_x, target_y = row[x_col], row[y_col]
        targets = {k: row[k] for k in [x_col, y_col] if k in METRIC_TO_PPA_LABEL}
        
        ctx = row_to_context(row, args.roadmap)
        opt_ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}
        opt_ctx["data_stacked_die_count"] = 1.0 # Force 2D designs
        
        design, ppa, pre_snap, snapped_ppa = opt.optimize(targets, opt_ctx, steps=args.opt_steps, verbose=args.verbose_opt)
        surr_x, surr_y = ppa.get(METRIC_TO_PPA_LABEL.get(x_col, "")), ppa.get(METRIC_TO_PPA_LABEL.get(y_col, ""))
        
        errs = [abs(pct_err(v, t)) for v, t in [(surr_x, target_x), (surr_y, target_y)] if v is not None]
        surr_mean_err = float(np.mean(errs)) if errs else float("nan")

        snapped_surr_x = snapped_ppa.get(METRIC_TO_PPA_LABEL.get(x_col, "")) if snapped_ppa else None
        snapped_surr_y = snapped_ppa.get(METRIC_TO_PPA_LABEL.get(y_col, "")) if snapped_ppa else None
        
        snap_errs = [abs(pct_err(v, t)) for v, t in [(snapped_surr_x, target_x), (snapped_surr_y, target_y)] if v is not None]
        snap_mean_err = float(np.mean(snap_errs)) if snap_errs else float("nan")

        x_pct, y_pct = float((df[x_col] <= target_x).mean() * 100), float((df[y_col] <= target_y).mean() * 100)

        # Optimized cell params (public names so they aren't stripped from CSV)
        opt_wn  = design.get("CellInput_SRAMCellNMOSWidth (F)", ctx["_wn"])
        opt_wp  = design.get("CellInput_SRAMCellPMOSWidth (F)", ctx["_wp"])
        opt_wac = design.get("CellInput_AccessCMOSWidth (F)",   ctx["_wac"])
        opt_rv  = design.get("CellInput_ReadVoltage (V)",        ctx["_read_voltage"])

        records.append({
            # --- metadata ---
            "is_original": False,
            "target_idx": i, "x_percentile": x_pct, "y_percentile": y_pct,
            "node_nm": int(row["process_node_nm"]),
            "target_x": target_x, "target_y": target_y, 
            "surr_x": surr_x, "surr_y": surr_y,
            "surr_err_x_pct": pct_err(surr_x, target_x) if surr_x is not None else None,
            "surr_err_y_pct": pct_err(surr_y, target_y) if surr_y is not None else None,
            "surr_mean_abs_err_pct": surr_mean_err,
            "post_snap_surr_x": snapped_surr_x, "post_snap_surr_y": snapped_surr_y,
            "post_snap_surr_err_x_pct": pct_err(snapped_surr_x, target_x) if snapped_surr_x is not None else None,
            "post_snap_surr_err_y_pct": pct_err(snapped_surr_y, target_y) if snapped_surr_y is not None else None,
            "post_snap_surr_mean_abs_err_pct": snap_mean_err,
            # --- original dataset inputs (the design that achieved the target PPA) ---
            "orig_capacity_kb":            row.get("capacity_kb"),
            "orig_word_width_bits":        row.get("word_width_bits"),
            "orig_associativity":          row.get("associativity"),
            "orig_data_stacked_die_count": row.get("data_stacked_die_count"),
            **{f"orig_{k}": row.get(k) for k in _LAYOUT_COLS},
            "orig_wn":  ctx["_wn"],  "orig_wp":  ctx["_wp"],
            "orig_wac": ctx["_wac"], "orig_rv":  ctx["_read_voltage"],
            # --- optimized inputs (what we re-injected into DESTINY) ---
            "capacity_kb": design.get("capacity_kb"), "word_width_bits": design.get("word_width_bits"),
            "associativity": design.get("associativity"), "data_stacked_die_count": design.get("data_stacked_die_count"),
            **{k: design.get(k) for k in _LAYOUT_COLS},
            "opt_wn": opt_wn, "opt_wp": opt_wp, "opt_wac": opt_wac, "opt_rv": opt_rv,
            # --- private: used internally, stripped before saving ---
            "_wn": opt_wn, "_wp": opt_wp, "_wac": opt_wac, "_read_voltage": opt_rv,
            "_temp": SRAM_TEMPERATURE_MAP.get(design.get("data_stacked_die_count", 1), ctx["_temp"]),
            # --- DESTINY validation results (filled in step 5) ---
            "destiny_x": None, "destiny_y": None, "destiny_err_x_pct": None, "destiny_err_y_pct": None, "destiny_mean_abs_err_pct": None,
            # --- pre-snap continuous values from the optimizer ---
            **{f"pre_snap_{k}": v for k, v in pre_snap.items()},
        })
        print(f"  [{i+1:3d}/{len(df_target)}] Target: {x_label.split()[0]}={target_x:.4g}{x_unit} (p{x_pct:.0f}), {y_label.split()[0]}={target_y:.4g}{y_unit} (p{y_pct:.0f})\n"
              f"         Pre-snap Surr:  {surr_x:.4g}, {surr_y:.4g}  (err={surr_mean_err:.1f}%)\n"
              f"         Post-snap Surr: {snapped_surr_x:.4g}, {snapped_surr_y:.4g}  (err={snap_mean_err:.1f}%)")
    return pd.DataFrame(records)

def _validate_top_designs(df_res, x_col, y_col, n_validate, args):
    """Run DESTINY validation on the lowest-surrogate-error designs."""
    if n_validate <= 0:
        print("\n[5/6] Skipping DESTINY validation (--validate-top 0)\n")
        return df_res

    print(f"\n[5/6] DESTINY validation for top-{n_validate} designs (ranked by surrogate error)...\n")
    for rank, ri in enumerate(df_res["surr_mean_abs_err_pct"].sort_values().head(n_validate).index, 1):
        row = df_res.loc[ri]
        ot = "ReadLatency"
        node_val = int(row.get("node_nm", args.node))
        
        print(f"  Validating design #{rank}  (target_idx={int(row.target_idx)}, surr_err={row.surr_mean_abs_err_pct:.1f}%)\n"
              f"    Node={node_val}nm  Cap={int(row.capacity_kb)}KB  WW={row.word_width_bits}  Assoc={row.associativity}  Stack={row.data_stacked_die_count}  OptTarget={ot}")

        layout = _layout_from_row(row)
        destiny_ppa = validate_and_capture(
            tech=args.tech, cap_kb=int(row.capacity_kb), ww=int(row.word_width_bits),
            assoc=int(row.associativity), stack=max(1, int(row.data_stacked_die_count)),
            temp=int(row["_temp"]), wn=float(row["_wn"]), wp=float(row["_wp"]), wac=float(row["_wac"]), read_voltage=float(row["_read_voltage"]),
            node=node_val, roadmap=args.roadmap, timeout=args.destiny_timeout, verbose=args.verbose_destiny, opt_target=ot,
            mux_sa=layout.get("data_mux_sense_amp"), mux_ol2=layout.get("data_mux_output_lev2"),
            act_mat_col=layout.get("data_num_active_mat_per_col"), act_mat_row=layout.get("data_num_active_mat_per_row"),
            act_sub_col=layout.get("data_num_active_subarray_per_col"), act_sub_row=layout.get("data_num_active_subarray_per_row"),
        )
        
        if destiny_ppa is None:
            print("    DESTINY failed\n"); continue

        dx, dy = destiny_ppa.get(x_col), destiny_ppa.get(y_col)
        dx_err = pct_err(dx, row.target_x) if dx is not None else None
        dy_err = pct_err(dy, row.target_y) if dy is not None else None
        d_mean = np.mean([abs(e) for e in [dx_err, dy_err] if e is not None])

        df_res.at[ri, "destiny_x"] = dx
        df_res.at[ri, "destiny_y"] = dy
        df_res.at[ri, "destiny_err_x_pct"] = dx_err
        df_res.at[ri, "destiny_err_y_pct"] = dy_err
        df_res.at[ri, "destiny_mean_abs_err_pct"] = d_mean

        xk, yk = x_col.split("_")[2], y_col.split("_")[2]
        print(f"    DESTINY -> {xk}={dx:.4g}, {yk}={dy:.4g}\n    Errors:  delta_{xk}={dx_err:+.1f}%  delta_{yk}={dy_err:+.1f}%  mean={d_mean:.1f}%\n")
    return df_res

def _plot_results(df_res, df, df_target, df_pareto_ref, x_col, y_col, x_label, y_label, args, s_errs, d_errs):
    """Generate and save benchmark plot."""
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    norm, _ = cap_colormap(df["capacity_kb"])
    
    # Background data points
    ax.scatter(df[x_col], df[y_col], c=df["capacity_kb"], norm=norm, cmap=plt.cm.viridis, s=14, alpha=0.15, linewidths=0, zorder=1, label="All data")

    # Pareto reference
    pf_alpha, pf_label = (0.85, "Pareto front (target)") if args.mode == "pareto" else (0.35, "Pareto front (ref)")
    sx_p, sy_p = pareto_step_line(df_pareto_ref[x_col].values, df_pareto_ref[y_col].values)
    ax.plot(sx_p, sy_p, color="#aaaaaa", lw=1.5 if args.mode == "pareto" else 1.0, ls="-" if args.mode == "pareto" else "--", zorder=2, alpha=pf_alpha)
    ax.scatter(df_pareto_ref[x_col], df_pareto_ref[y_col], c=df_pareto_ref["capacity_kb"], norm=norm, cmap=plt.cm.viridis, s=55 if args.mode == "pareto" else 28, marker="D", edgecolors="k", linewidths=0.5, zorder=3, alpha=pf_alpha, label=pf_label)

    # Median targets
    if args.mode == "median":
        ax.scatter(df_target[x_col], df_target[y_col], facecolors="none", s=70, marker="s", edgecolors="k", linewidths=1.5, zorder=4, label="Median sample (target)")

    # Surrogate predictions
    mask_surr = df_res["surr_x"].notna() & df_res["surr_y"].notna()
    ax.scatter(df_res.loc[mask_surr, "surr_x"], df_res.loc[mask_surr, "surr_y"], c="#ffa657", s=60, marker="o", edgecolors="k", linewidths=0.5, zorder=5, alpha=0.85, label="Surrogate prediction")
    for _, r in df_res[mask_surr].iterrows(): ax.plot([r.target_x, r.surr_x], [r.target_y, r.surr_y], color="#ffa657", lw=0.8, alpha=0.5, zorder=3)

    # DESTINY validated points
    mask_dest = df_res["destiny_x"].notna() & df_res["destiny_y"].notna()
    if mask_dest.any():
        ax.scatter(df_res.loc[mask_dest, "destiny_x"], df_res.loc[mask_dest, "destiny_y"], c="#3fb950", s=90, marker="*", edgecolors="k", linewidths=0.5, zorder=6, label="DESTINY validated")
        for _, r in df_res[mask_dest].iterrows(): ax.plot([r.target_x, r.destiny_x], [r.target_y, r.destiny_y], color="#3fb950", lw=1.0, alpha=0.6, zorder=4, linestyle="--")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"{x_label} (log scale)", fontsize=10); ax.set_ylabel(f"{y_label} (log scale)", fontsize=10)
    format_log_axis(ax, axis="both")

    subtitle = f"{len(df_target)} targets, {args.n_bins} bins" if args.mode == "median" else f"{len(df_target)} Pareto points"
    ax.set_title(f"Inverse Optimizer vs {args.mode.capitalize()} Points  [{subtitle}]\n{args.tech} | {args.node}nm | {args.roadmap}", fontsize=12, fontweight="bold", pad=10)

    post_s_errs = df_res["post_snap_surr_mean_abs_err_pct"].dropna()
    ann_lines = [
        f"Pre-snap Surr  mean |err|: {s_errs.mean():.1f}% (n={len(s_errs)})",
        f"Post-snap Surr mean |err|: {post_s_errs.mean():.1f}% (n={len(post_s_errs)})"
    ]
    if len(d_errs) > 0: ann_lines.append(f"DESTINY        mean |err|: {d_errs.mean():.1f}% (n={len(d_errs)})")
    ax.text(0.02, 0.97, "\n".join(ann_lines), transform=ax.transAxes, fontsize=8.5, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa", alpha=0.9))

    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="lower right", fontsize=9); add_cap_colorbar(fig, [ax], norm)
    png_path = os.path.join(args.output_dir, f"benchmark_{args.mode}_{args.tech}_{args.node}nm_{args.roadmap}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"           Plot saved -> {png_path}\n")

# -- Main ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark InverseOptimizer against Pareto or median points.")
    p.add_argument("--tech", default="SRAM")
    p.add_argument("--node", type=int, default=32, help="Process node [nm]")
    p.add_argument("--roadmap", default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--mode", default="pareto", choices=["pareto", "median"])
    p.add_argument("--n-bins", type=int, default=20, help="[median] x-quantile bins")
    p.add_argument("--x-metric", default="cache_hit_latency_ns", choices=list(METRIC_META))
    p.add_argument("--y-metric", default="cache_area_mm2", choices=list(METRIC_META))
    p.add_argument("--validate-top", type=int, default=20, help="Top-N designs to DESTINY-validate")
    p.add_argument("--opt-steps", type=int, default=400, help="Gradient steps per optimization")
    p.add_argument("--destiny-timeout", type=int, default=300, help="Timeout [s] per DESTINY call")
    p.add_argument("--output-dir", default="benchmark_results")
    p.add_argument("--verbose-destiny", action="store_true", help="Print DESTINY cell/cfg and full stdout on every call")
    p.add_argument("--verbose-opt",     action="store_true", help="Print pre/post-snap parameter table for every optimized design")
    p.add_argument("--max-targets", type=int, default=None, help="Max target designs to optimize (None for all)")
    p.add_argument("--feasibility", action="store_true", help="Use feasibility classifier to penalize impossible designs")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    x_col, y_col = args.x_metric, args.y_metric
    x_label, x_unit = METRIC_META[x_col]["label"], METRIC_META[x_col]["unit"]
    y_label, y_unit = METRIC_META[y_col]["label"], METRIC_META[y_col]["unit"]
    data_csv = os.path.join("pareto", args.tech, f"{args.tech}_{'pareto' if args.mode == 'pareto' else 'full_data'}.csv")

    print(f"\n{'='*70}\n  Benchmark [{args.mode.capitalize()}]: {args.tech} | {args.node}nm | {args.roadmap}\n  X: {x_label}   Y: {y_label}   Dataset: {data_csv}\n{'='*70}\n")

    df = _load_and_filter_data(args, data_csv, x_col, y_col)

    print(f"\n[2/6] {'Extracting Pareto front' if args.mode == 'pareto' else f'Extracting median points ({args.n_bins} x-bins)'} on {x_col} x {y_col}")
    if args.mode == "pareto":
        df_target = df_pareto_ref = select_pareto(df, x_col, y_col)
    else:
        df_target = select_median(df, x_col, y_col, n_bins=args.n_bins)
        df_pareto_ref = df[pareto_frontier_2d(df[x_col].values, df[y_col].values)]

    if args.max_targets is not None: df_target = df_target.head(args.max_targets)

    print(f"\n[3/6] Loading InverseOptimizer (tech={args.tech}, feasibility={args.feasibility})")
    opt = InverseOptimizer(args.tech, use_feasibility=args.feasibility)

    print(f"\n[4/6] Running inverse optimization for {len(df_target)} targets ({args.opt_steps} steps each)...\n")
    df_res = _run_optimization_sweep(df_target, df, opt, x_col, y_col, x_label, y_label, x_unit, y_unit, args)

    df_res = _validate_top_designs(df_res, x_col, y_col, min(args.validate_top, len(df_res)), args)

    s_errs = df_res["surr_mean_abs_err_pct"].dropna()
    post_s_errs = df_res["post_snap_surr_mean_abs_err_pct"].dropna()
    d_errs = df_res["destiny_mean_abs_err_pct"].dropna()
    csv_path = os.path.join(args.output_dir, f"benchmark_{args.mode}_{args.tech}_{args.node}nm_{args.roadmap}.csv")
    df_res[[c for c in df_res.columns if not c.startswith("_")]].to_csv(csv_path, index=False)
    
    print(f"[6/6] Results saved -> {csv_path}\n"
          f"      Pre-snap Surr  mean |err|: {s_errs.mean():.2f}%  (median {s_errs.median():.2f}%,  n={len(s_errs)})\n"
          f"      Post-snap Surr mean |err|: {post_s_errs.mean():.2f}%  (median {post_s_errs.median():.2f}%,  n={len(post_s_errs)})")
    if len(d_errs) > 0: print(f"      DESTINY        mean |err|: {d_errs.mean():.2f}%  (median {d_errs.median():.2f}%,  n={len(d_errs)})")

    _plot_results(df_res, df, df_target, df_pareto_ref, x_col, y_col, x_label, y_label, args, s_errs, d_errs)

if __name__ == "__main__":
    main()
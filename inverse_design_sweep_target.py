#!/usr/bin/env python3
"""Runs one configuration + produces one csv + plot"""

import os, sys, argparse, warnings, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_design_gumbel_target import InverseOptimizerGumbel
from destiny_utils import (
    cap_colormap,
    format_log_axis,
    add_cap_colorbar,
    setup_opt_dirs,
    METRIC_META,
    METRIC_TO_PPA_LABEL,
    TECH_SKIP_TARGETS,
    get_active_targets,
)
from inverse_design_utils import (
    validate_and_capture,
    row_to_context,
    pct_err,
    layout_from_row as _layout_from_row,
    BENCHING_LAYOUT_COLS,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Temperature co-varies with stack count (from run_exploration.py).
SRAM_TEMPERATURE_MAP = {1: 300, 2: 363, 4: 380}

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
            "is_original": False, "target_idx": i, "node_nm": int(row["process_node_nm"]), "device_roadmap": args.roadmap, "method": "gumbel",
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
            cap_kb=int(row.capacity_kb), ww=int(row.word_width_bits),
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
            cap_kb=int(row.orig_capacity_kb), ww=int(row.orig_word_width_bits),
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

def _compute_and_report_trend_rank(args, base_dir):
    """Aggregates all method sweep run records to rank optimization algorithms by their mean absolute error percentage across all active hardware targets."""
    csv_files = glob.glob(os.path.join(base_dir, "destiny_files", "*", "benchmark_pareto_*.csv"))
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
        f"{args.tech} | {args.node}nm | {args.roadmap} | method=gumbel",
        fontsize=11, fontweight="bold"
    )
    fig.savefig(
        os.path.join(
            plots_dir,
            f"benchmark_trajectory_{args.tech}_{args.node}nm_{args.roadmap}_gumbel.png"
        ),
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Multi-metric cache inverse sizing benchmark engine.")
    p.add_argument("--tech", default="SRAM")
    p.add_argument("--node", type=int, default=32, help="Process node nm constraint")
    p.add_argument("--roadmap", default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--metrics", nargs="+", default=None, choices=list(METRIC_META), help="PPA objectives to size for (default: all active metrics for --tech)")
    p.add_argument("--opt-steps", type=int, default=120, help="Sizing optimization gradient steps")
    p.add_argument("--validate-top", type=int, default=5, help="Number of solved layouts to validate with C++ compiler")
    p.add_argument("--destiny-timeout", type=int, default=600, help="TIMEOUT for physical compilation subprocesses")
    p.add_argument("--output-dir", default="opt_target")
    p.add_argument("--max-targets", type=int, default=5, help="Maximum index constraints to evaluate from library")
    p.add_argument("--capacity-kb", type=float, default=None, help="Filter the target vectors to a specific capacity (in KB)")
    p.add_argument("--run-id", default=None, help="Optional run ID for locating the output directory under opt_target/")
    p.add_argument("--verbose-destiny", action="store_true", help="Print compiler outputs directly to console")
    p.add_argument("--verbose-opt", action="store_true", help="Print tabular design updates during steps")
    args = p.parse_args()
    if args.metrics is None:
        args.metrics = get_active_targets(args.tech)

    base_dir, destiny_logs_dir, plots_dir = setup_opt_dirs("opt_target", args.run_id)

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

    print(f"\n{'='*80}\n  Runner Sweeper: {args.tech} | Node={args.node}nm | Roadmap={args.roadmap} | Method=gumbel\n  Targets: {args.metrics}\n{'='*80}\n")

    ppa_data_frame = _load_and_filter_data(args, data_csv)
    target_vectors = ppa_data_frame.head(args.max_targets)

    print(f"\n[2/5] Initializing InverseOptimizer (Gumbel)")
    sizing_optimizer = InverseOptimizerGumbel(args.tech)

    print(f"\n[3/5] Solving layouts across {len(target_vectors)} target PPA metrics...")
    ppa_results = _run_optimization_sweep(target_vectors, ppa_data_frame, sizing_optimizer, args)

    print(f"\n[4/5] Running physical validation for top-{args.validate_top} optimal configurations...")
    ppa_results = _validate_top_designs(ppa_results, min(args.validate_top, len(ppa_results)), args, destiny_logs_dir)

    csv_path = os.path.join(destiny_logs_dir, f"benchmark_pareto_{args.tech}_{args.node}nm_{args.roadmap}_gumbel.csv")
    ppa_results[[c for c in ppa_results.columns if not c.startswith("_")]].to_csv(csv_path, index=False)
    print(f"[5/5] Sized results successfully logged -> {csv_path}")

    _compute_and_report_trend_rank(args, base_dir)
    _plot_results_side_effect(ppa_results, ppa_data_frame, target_vectors, args, plots_dir)

if __name__ == "__main__":
    main()
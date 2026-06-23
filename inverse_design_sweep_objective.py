#!/usr/bin/env python3
"""Runs objective based sweep optimization across baseline designs + produces CSVs + plots"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from inverse_design_gumbel_objective import InverseOptimizerGumbel
from destiny_utils import (
    pareto_frontier_nd,
    format_log_axis,
    setup_opt_dirs,
    validate_cache_geometry,
    METRIC_META,
    METRIC_TO_PPA_LABEL,
    TECH_SKIP_TARGETS,
    get_active_targets,
)
from inverse_design_utils import (
    validate_and_capture,
    row_to_context,
    layout_from_row as _layout_from_row,
    BENCHING_LAYOUT_COLS,
)


warnings.filterwarnings("ignore", category=UserWarning)

# Temperature co-varies with stack count (from run_exploration.py).
SRAM_TEMPERATURE_MAP = {1: 300, 2: 363, 4: 380}


def _load_and_filter_data(args, data_csv):
    """Load and filter the PPA data sweep library."""
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
    """Run optimization over all selected Pareto target vectors."""
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

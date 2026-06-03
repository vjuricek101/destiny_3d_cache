#!/usr/bin/env python3
"""Wrapper orchestrating combinatorial Cartesian sweep configurations"""

import os
import sys
import argparse
import itertools
import subprocess
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from destiny_utils import TECH_SKIP_TARGETS, get_active_targets

def get_default_nodes_roadmaps(tech):
    """Parses raw datasets to identify available hardware process nodes and device roadmaps."""
    data_csv = Path("pareto") / tech / f"{tech}_full_data.csv"
    if not data_csv.exists():
        sys.exit(f"ERROR: Dataset not found at {data_csv}")
    
    ppa_data_frame = pd.read_csv(data_csv)
    df_tech = ppa_data_frame[ppa_data_frame["mem_cell_type"].str.upper() == tech.upper()]
    return (
        sorted(df_tech["process_node_nm"].dropna().unique().astype(int).tolist()),
        sorted(df_tech["device_roadmap"].dropna().unique().tolist())
    )

def main():
    p = argparse.ArgumentParser(description="Combinatorial sweep wrapper for hardware design space optimization.")
    p.add_argument("--tech", default="SRAM", help="Memory cell technology (default: SRAM)")
    p.add_argument("--nodes", type=int, nargs="+", default=None, help="Process nodes [nm] to sweep")
    p.add_argument("--roadmaps", nargs="+", default=None, choices=["HP", "LOP", "LSTP"], help="Device roadmaps to sweep")
    p.add_argument("--variants", nargs="+", default=["baseline", "ste", "gumbel"], choices=["baseline", "ste", "gumbel"], help="Optimizer engines to sweep")
    p.add_argument("--metrics", nargs="+", default=None, help="PPA metrics to target (default: all active metrics for --tech)")
    p.add_argument("--opt-steps", type=int, default=120, help="Optimization steps per run")
    p.add_argument("--validate-top", type=int, default=5, help="Number of solved layouts to validate")
    p.add_argument("--destiny-timeout", type=int, default=600, help="Timeout for DESTINY validator calls")
    p.add_argument("--max-targets", type=int, default=5, help="Max target designs to evaluate")
    p.add_argument("--verbose-destiny", action="store_true", help="Print simulator outputs to console")
    p.add_argument("--verbose-opt", action="store_true", help="Print tabular design updates during steps")
    p.add_argument("--output-dir", default="multi_sweep_results", help="Directory to write all CSV outputs")
    args = p.parse_args()
    if args.metrics is None:
        args.metrics = get_active_targets(args.tech)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter structurally-zero metrics before the sweep grid is built so no subprocess is wasted.
    skip = set(TECH_SKIP_TARGETS.get(args.tech, []))
    rejected = [m for m in args.metrics if m in skip]
    if rejected:
        print(f"[WARN] Dropping structurally-zero metrics for {args.tech}: {rejected}")
        args.metrics = [m for m in args.metrics if m not in skip]
    if not args.metrics:
        sys.exit(f"ERROR: All requested metrics are structurally zero for {args.tech}.")

    nodes, roadmaps = get_default_nodes_roadmaps(args.tech)
    nodes = args.nodes if args.nodes is not None else nodes
    roadmaps = args.roadmaps if args.roadmaps is not None else roadmaps

    sweep_grid = list(itertools.product(nodes, roadmaps, args.variants))
    total_runs = len(sweep_grid)

    print("=" * 80)
    print(f"  Multi-Sweep Orchestrator: {args.tech} Cache Optimization Grid")
    print(f"  Nodes      : {nodes}")
    print(f"  Roadmaps   : {roadmaps}")
    print(f"  Variants   : {args.variants}")
    print(f"  Target PPA : {args.metrics}")
    print(f"  Grid Size  : {len(nodes)} nodes x {len(roadmaps)} roadmaps x {len(args.variants)} variants = {total_runs} runs")
    print(f"  Output Dir : {out_dir.resolve()}")
    print("=" * 80)

    n_done = n_failed = 0

    # Process boundaries isolate PyTorch GPU contexts and prevent memory accumulation across grid configurations
    for run_idx, (node, roadmap, variant) in enumerate(sweep_grid, 1):
        print(f"\n[{run_idx}/{total_runs}] Initiating sweep run: Node={node}nm | Roadmap={roadmap} | Variant={variant}")
        
        cmd = [
            "python3", "inverse_design_sweep.py",
            "--tech", args.tech,
            "--node", str(node),
            "--roadmap", roadmap,
            "--method", variant,
            "--metrics", *args.metrics,
            "--opt-steps", str(args.opt_steps),
            "--destiny-timeout", str(args.destiny_timeout),
            "--output-dir", str(out_dir),
            "--validate-top", str(args.validate_top),
            "--max-targets", str(args.max_targets),
        ]
        if args.verbose_destiny: cmd.append("--verbose-destiny")
        if args.verbose_opt: cmd.append("--verbose-opt")

        try:
            subprocess.run(cmd, check=True)
            n_done += 1
            print(f"  [success] Sweep Node={node}nm | {roadmap} | {variant} completed.")
        except subprocess.CalledProcessError as e:
            n_failed += 1
            print(f"  [error] Subprocess failed: {e}")

# Analysis & Ranking

    print("\n" + "=" * 80)
    print("  Aggregating Sweep Results & Computing Trend Ranking")
    print("=" * 80)

    csv_pattern = str(out_dir / f"benchmark_pareto_{args.tech}_*nm_*_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print("  [warning] No benchmark output CSVs found in output directory. Skipping trend analysis.")
        return

    aggregated_ppa_results = []
    for f in csv_files:
        try:
            ppa_results = pd.read_csv(f)
            if ppa_results.empty: continue
            
            parts = Path(f).stem.split("_")
            if len(parts) >= 6:
                ppa_results["method"] = parts[-1]
                ppa_results["node"] = int(parts[-3].replace("nm", ""))
                ppa_results["roadmap"] = parts[-2]
                aggregated_ppa_results.append(ppa_results)
        except Exception as e:
            print(f"  [warning] Error reading {f}: {e}")

    if not aggregated_ppa_results:
        print("  [warning] Could not load any valid dataset rows. Skipping trend analysis.")
        return

    aggregated_ppa_dataframe = pd.concat(aggregated_ppa_results, ignore_index=True)

    group_cols = ["method"]
    agg_funcs = {
        "surr_mean_abs_err_pct": ["mean", "median", "count"],
        "post_snap_surr_mean_abs_err_pct": ["mean", "median"],
    }
    
    if "destiny_mean_abs_err_pct" in aggregated_ppa_dataframe.columns:
        agg_funcs["destiny_mean_abs_err_pct"] = ["mean", "median"]

    performance_ranking_dataframe = aggregated_ppa_dataframe.groupby(group_cols).agg(agg_funcs)
    performance_ranking_dataframe.columns = [f"{col[0]}_{col[1]}" for col in performance_ranking_dataframe.columns]
    performance_ranking_dataframe = performance_ranking_dataframe.reset_index()
    
    # Calculate rank correlations per method aggregated over all individual sweeps
    k0 = args.metrics[0]
    target_col = f"target_{k0}"
    
    corr_records = []
    for method in aggregated_ppa_dataframe["method"].unique():
        df_method = aggregated_ppa_dataframe[aggregated_ppa_dataframe["method"] == method]
        pred_col = f"destiny_{k0}" if (f"destiny_{k0}" in df_method.columns and df_method[f"destiny_{k0}"].notna().any()) else f"post_snap_surr_{k0}"
        
        taus = []
        rhos = []
        if "node" in df_method.columns and "roadmap" in df_method.columns:
            for (node, roadmap), df_run in df_method.groupby(["node", "roadmap"]):
                if target_col in df_run.columns and pred_col in df_run.columns:
                    df_valid = df_run.dropna(subset=[target_col, pred_col])
                    if len(df_valid) >= 3:
                        df_valid = df_valid.sort_values(by=target_col)
                        tau = df_valid[target_col].corr(df_valid[pred_col], method="kendall")
                        rho = df_valid[target_col].corr(df_valid[pred_col], method="spearman")
                        if not np.isnan(tau): taus.append(tau)
                        if not np.isnan(rho): rhos.append(rho)
        
        mean_tau = np.mean(taus) if taus else float("nan")
        mean_rho = np.mean(rhos) if rhos else float("nan")
        
        corr_records.append({
            "method": method,
            "kendall_tau": mean_tau,
            "spearman_rho": mean_rho
        })
    df_corrs = pd.DataFrame(corr_records)
    
    performance_ranking_dataframe = pd.merge(performance_ranking_dataframe, df_corrs, on="method", how="left")
    
    # Ranks physical simulation compiler error first; falls back to model snap error if validation is disabled
    rank_metric = "destiny_mean_abs_err_pct_mean" if "destiny_mean_abs_err_pct_mean" in performance_ranking_dataframe.columns else "post_snap_surr_mean_abs_err_pct_mean"
    
    performance_ranking_dataframe = performance_ranking_dataframe.sort_values(by=rank_metric, ascending=True)
    performance_ranking_dataframe["trend_rank"] = range(1, len(performance_ranking_dataframe) + 1)

    # Format correlation values cleanly for console printout
    print_df = performance_ranking_dataframe.copy()
    print_df["kendall_tau"] = print_df["kendall_tau"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    print_df["spearman_rho"] = print_df["spearman_rho"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")

    print("\n### Optimization Engine Trend Ranking (Sorted by Validation Accuracy):")
    sub_cols = ["trend_rank", "method", "surr_mean_abs_err_pct_count", "surr_mean_abs_err_pct_mean", "post_snap_surr_mean_abs_err_pct_mean", "kendall_tau", "spearman_rho"] + (["destiny_mean_abs_err_pct_mean"] if "destiny_mean_abs_err_pct_mean" in print_df.columns else [])
    sub_df = print_df[sub_cols]
    try:
        print(sub_df.to_markdown(index=False))
    except ImportError:
        print(sub_df.to_string(index=False))

    summary_path = out_dir / "multi_sweep_trend_summary.csv"
    performance_ranking_dataframe.to_csv(summary_path, index=False)
    print(f"\n  Summary trend ranking report successfully written to: {summary_path}")

    print("\n" + "=" * 80)
    print(f"  Sweep Grid Complete: {n_done} runs completed, {n_failed} runs failed.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

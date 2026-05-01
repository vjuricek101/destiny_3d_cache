#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Pareto Algorithms

def is_pareto_efficient(costs):
    """O(n^2) Pareto frontier for N objectives."""
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]: continue
        others = costs[is_efficient]
        dominated = (np.all(others <= costs[i], axis=1) & np.any(others < costs[i], axis=1))
        efficient_indices = np.where(is_efficient)[0]
        self_pos = np.where(efficient_indices == i)[0][0]
        dominated[self_pos] = False
        if np.any(dominated): is_efficient[i] = False
    return is_efficient

# Data Loading

def load_sim_csv(results_path, tech, is_arch=False):
    """Loads a single simulation CSV."""
    try:
        df = pd.read_csv(results_path)
        if df.empty: return None
        return df
    except Exception as e:
        return None

# Orchestration

def process_results(tech, is_arch, only_full):
    suffix = "_arch" if is_arch else ""
    res_dir = Path(f"exploration_results/{tech}{suffix}")
    if not res_dir.exists(): return
    
    csv_files = list(res_dir.glob("*.csv"))
    if not csv_files: return

    print(f"Aggregating {tech}{suffix} ({len(csv_files)} files)...")
    
    all_dfs = []
    chunk_size = 5000
    for i in range(0, len(csv_files), chunk_size):
        chunk = csv_files[i : i + chunk_size]
        dfs = [load_sim_csv(str(f), tech, is_arch) for f in chunk]
        valid_dfs = [d for d in dfs if d is not None]
        if valid_dfs:
            all_dfs.append(pd.concat(valid_dfs, ignore_index=True))
        print(f"PROGRESS: Processed {min(i + chunk_size, len(csv_files))}/{len(csv_files)}...")

    if not all_dfs:
        print(f"ERROR: No valid data found in {res_dir}")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    out_dir = Path(f"pareto/{tech}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_dir / f"{tech}{suffix}_full_data.csv", index=False)
    
    if only_full: return

    # Calculation logic for filters
    ppa_cols = ["cache_hit_latency_ns", "cache_area_mm2", "cache_hit_energy_nJ", "cache_leakage_mW"]
    pareto_frames = []
    
    print(f"  Calculating Pareto frontiers...")
    for cap, group in full_df.groupby('capacity_kb'):
        costs = group[ppa_cols].values
        p_df = group[is_pareto_efficient(costs)]
        p_df.to_csv(out_dir / f"{tech}{suffix}_cap_{str(cap).replace('.','_')}_pareto.csv", index=False)
        pareto_frames.append(p_df)
    
    if pareto_frames:
        pd.concat(pareto_frames).to_csv(out_dir / f"{tech}{suffix}_pareto.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Pareto Analysis Core")
    parser.add_argument("--type", default="ALL", choices=["SRAM", "RRAM", "eDRAM", "ALL"])
    parser.add_argument("--arch", action="store_true", help="Process architectural sweep results (_arch)")
    parser.add_argument("--only-full", action="store_true", help="Only generate the full merged CSV, skip Pareto filtering")
    args = parser.parse_args()
    
    techs = ["SRAM", "RRAM", "eDRAM"] if args.type == "ALL" else [args.type]
    for t in techs:
        process_results(t, args.arch, args.only_full)
    
    # Unified Global Merge
    all_p = [pd.read_csv(f) for f in Path("pareto").glob("*/*_pareto.csv") if "cap" not in f.name]
    if all_p:
        print("\nCreating final unified Pareto CSV...")
        pd.concat(all_p, ignore_index=True).drop_duplicates().to_csv("pareto/pareto.csv", index=False)
    print("Done.")

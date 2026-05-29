#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# -- Pareto Algorithm ----------------------------------------------------------

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

# -- Data Loading --------------------------------------------------------------

def load_sim_csv(results_path, tech):
    """Loads a single simulation CSV."""
    try:
        df = pd.read_csv(results_path)
        if df.empty: return None
        return df
    except Exception:
        return None

# -- Pareto Aggregation --------------------------------------------------------

def process_results(tech, only_full):
    res_dir = Path(f"exploration_results/{tech}")
    if not res_dir.exists(): return

    csv_files = list(res_dir.glob("*.csv"))
    if not csv_files: return

    print(f"Aggregating {tech} ({len(csv_files)} files)...")

    all_dfs = []
    chunk_size = 5000
    for i in range(0, len(csv_files), chunk_size):
        chunk = csv_files[i : i + chunk_size]
        dfs = [load_sim_csv(str(f), tech) for f in chunk]
        valid_dfs = [d for d in dfs if d is not None]
        if valid_dfs:
            all_dfs.append(pd.concat(valid_dfs, ignore_index=True))
        print(f"PROGRESS: Processed {min(i + chunk_size, len(csv_files))}/{len(csv_files)}...")

    if not all_dfs:
        print(f"ERROR: No valid data found in {res_dir}")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    out_dir = Path(f"pareto/{tech}")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_dir / f"{tech}_full_data.csv", index=False)

    if only_full: return

    ppa_cols = [
        "cache_area_mm2",
        "cache_hit_latency_ns",
        "cache_write_latency_ns",
        "cache_refresh_latency_ns",
        "cache_hit_energy_nJ",
        "cache_write_energy_nJ",
        "cache_refresh_energy_nJ",
        "cache_leakage_mW",
    ]
    pareto_df = full_df

    pareto_frames = []

    print(f"  Calculating Pareto frontiers...")
    for cap, group in pareto_df.groupby("capacity_kb"):
        costs = group[ppa_cols].values
        p_df = group[is_pareto_efficient(costs)]
        p_df.to_csv(out_dir / f"{tech}_cap_{str(cap).replace('.','_')}_pareto.csv", index=False)
        pareto_frames.append(p_df)

    if pareto_frames:
        pd.concat(pareto_frames).to_csv(out_dir / f"{tech}_pareto.csv", index=False)

# -- Feasibility Dataset -------------------------------------------------------

def process_feasibility(tech, output_dir="pareto"):
    """Merge valid simulation rows (is_valid=1) with failed rows (is_valid=0)
    into a single labeled CSV for feasibility classifier training.

    Sources:
      - exploration_results/<tech>/*.csv   -> is_valid=1
      - failed_exploration/<tech>/<tech>_failed.csv -> is_valid=0
    Output:
      - pareto/<tech>/<tech>_feasibility.csv
    """
    print(f"\n{'-' * 60}")
    print(f"  Building feasibility dataset for {tech}")
    print(f"{'-' * 60}")

    # Success rows -- one row per run (rows within a CSV differ only in opt_target)
    valid_dir     = f"exploration_results/{tech}"
    success_files = sorted(glob(f"{valid_dir}/*.csv")) if os.path.isdir(valid_dir) else []

    if success_files:
        chunks, skipped = [], 0
        for path in success_files:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"  [warn] Could not read {os.path.basename(path)}: {e}")
                skipped += 1
                continue
            if df.empty:
                skipped += 1
                continue
            # All OptimizationTargets kept
            chunks.append(df)
        df_valid = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        df_valid["is_valid"] = 1
        print(f"  Loaded {len(df_valid):>6} valid rows  ({skipped} files skipped)")
    else:
        df_valid = pd.DataFrame()
        print(f"  [warn] No success CSVs found in {valid_dir}")

    # Failure rows
    failed_path = f"failed_exploration/{tech}/{tech}_failed.csv"
    if os.path.exists(failed_path):
        df_failed = pd.read_csv(failed_path)
        print(f"  Loaded {len(df_failed):>6} failed rows  from {failed_path}")
    else:
        df_failed = pd.DataFrame()
        print(f"  [info] No failure CSV found at {failed_path} -- run sweep first")

    if df_valid.empty and df_failed.empty:
        print(f"  [skip] No data for {tech}")
        return

    df_all = pd.concat([df_valid, df_failed], ignore_index=True)

    vc       = df_all["is_valid"].value_counts()
    n_valid  = vc.get(1, 0)
    n_failed = vc.get(0, 0)
    pct_fail = 100.0 * n_failed / len(df_all) if len(df_all) else 0
    print(f"  Class balance:  valid={n_valid}  failed={n_failed}  "
          f"(failure rate={pct_fail:.1f}%)")

    out_dir  = Path(output_dir) / tech
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tech}_feasibility.csv"
    df_all.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}  ({len(df_all)} rows, {df_all.shape[1]} cols)")

# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pareto analysis + feasibility dataset builder")
    parser.add_argument("--type",        default="ALL", choices=["SRAM", "RRAM", "eDRAM", "ALL"])
    parser.add_argument("--only-full",   action="store_true", help="Skip Pareto filtering; write full_data CSV only")
    parser.add_argument("--output-dir",  default="pareto",    help="Root output directory (default: pareto/)")
    args = parser.parse_args()

    techs = ["SRAM", "RRAM", "eDRAM"] if args.type == "ALL" else [args.type]

    for t in techs:
        process_results(t, args.only_full)
        process_feasibility(t, output_dir=args.output_dir)

    # Unified global Pareto CSV
    all_p = [pd.read_csv(f) for f in Path("pareto").glob("*/*_pareto.csv") if "cap" not in f.name]
    if all_p:
        print("\nCreating final unified Pareto CSV...")
        pd.concat(all_p, ignore_index=True).drop_duplicates().to_csv("pareto/pareto.csv", index=False)
    print("Done.")

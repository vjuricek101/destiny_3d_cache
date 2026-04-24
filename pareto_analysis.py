#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Pareto Algorithms ──────────────────────────────────────────────────────────

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

# ── Utility Functions ─────────────────────────────────────────────────────────

def parse_cell_file(filepath):
    """Extract CellInput_ parameters from .cell file."""
    params = {}
    if not os.path.exists(filepath): return params
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('-'): continue
            parts = line[1:].split(':', 1)
            if len(parts) == 2:
                params[f"CellInput_{parts[0].strip()}"] = pd.to_numeric(parts[1].strip(), errors='ignore')
    return params

def parse_metadata(stem, tech, is_arch=False):
    """Extracts swept parameters from simulation output filenames."""
    meta = {'memory_technology': tech}
    
    # Common mappings
    mappings = {'cap_kb': r'_cap_(\d+)', 'word_width': r'_ww(\d+)', 
                'associativity': r'_a(\d+)', 'stacked_die_count': r'_s(\d+)', 
                'roadmap': r'_rm_([A-Z]+)'}
    
    # Specific mappings
    if is_arch:
        mappings['Temperature (K)'] = r'_t(\d+)'
    else:
        mappings['variant_id'] = r'variant_(\d+)'
        # For standard runs, temperature is derived from stack count (Level 4)
        m = re.search(r'_s(\d+)', stem)
        if m:
            temp_map = {"SRAM": {1: 350, 2: 363, 4: 380}, "eDRAM": {1: 350, 2: 363, 4: 380}, "RRAM": {1: 313, 2: 333, 4: 358}}
            stack = int(m.group(1))
            if tech in temp_map and stack in temp_map[tech]:
                meta['Temperature (K)'] = temp_map[tech][stack]

    for key, pattern in mappings.items():
        m = re.search(pattern, stem)
        if m: meta[key] = m.group(1)
        
    if 'cap_kb' in meta: meta['capacity_mb'] = float(meta['cap_kb']) / 1024.0
    return meta

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_sim_csv(results_path, tech, is_arch=False):
    """Loads a single simulation CSV and attaches metadata/cell physics."""
    try:
        df_raw = pd.read_csv(results_path, header=None, skipinitialspace=True)
        is_cache = len(df_raw.columns) >= 90
        cols = [1,2,6,10] if is_cache else [24,32,35,38]
        
        df = df_raw.iloc[:, cols].copy()
        df.columns = ["Cache Area (mm^2)", "Cache Hit Latency (ns)", "Cache Hit Energy (nJ)", "Cache Leakage Power (mW)"]
        if not is_cache: df.iloc[:, 2] /= 1000.0 # pJ -> nJ
        
        meta = parse_metadata(Path(results_path).stem, tech, is_arch)
        for k, v in meta.items(): df[k] = v
        
        # Resolve cell physics path
        m = re.search(r'_n(\d+)', results_path)
        node = m.group(1) if m else "32"
        
        if is_arch:
            cell_path = f"synthetic_cells/{tech}_arch/arch_variant_nominal_n{node}.cell"
        else:
            cell_path = f"synthetic_cells/{tech}/synthetic_variant_{meta.get('variant_id')}_n{node}.cell"
            
        for k, v in parse_cell_file(cell_path).items(): df[k] = v
        return df
    except Exception:
        return None

# ── Orchestration ─────────────────────────────────────────────────────────────

def process_results(tech, is_arch, only_full):
    suffix = "_arch" if is_arch else ""
    res_dir = Path(f"exploration_results/{tech}{suffix}")
    if not res_dir.exists(): return
    
    csv_files = list(res_dir.glob("*.csv"))
    if not csv_files: return

    print(f"Aggregating {tech}{suffix} ({len(csv_files)} files)...")
    dfs = [load_sim_csv(str(f), tech, is_arch) for f in csv_files]
    full_df = pd.concat([d for d in dfs if d is not None], ignore_index=True)
    
    out_dir = Path(f"pareto/{tech}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_dir / f"{tech}{suffix}_full_data.csv", index=False)
    
    if only_full: return

    # Calculation logic for filters
    ppa_cols = ["Cache Hit Latency (ns)", "Cache Area (mm^2)"]
    pareto_frames = []
    
    print(f"  Calculating Pareto frontiers...")
    for cap, group in full_df.groupby('capacity_mb'):
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

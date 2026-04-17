import os
import pandas as pd
import numpy as np
import argparse

def _pareto_2d_fast(costs):
    """O(n log n) Pareto frontier for 2 objectives."""
    n = costs.shape[0]
    if n == 0: return np.array([], dtype=bool)
    sort_idx = np.lexsort((costs[:, 1], costs[:, 0]))
    sorted_costs = costs[sort_idx]
    is_efficient_sorted = np.zeros(n, dtype=bool)
    min_obj1 = np.inf
    for i in range(n):
        if sorted_costs[i, 1] < min_obj1:
            is_efficient_sorted[i] = True
            min_obj1 = sorted_costs[i, 1]
    is_efficient = np.zeros(n, dtype=bool)
    is_efficient[sort_idx] = is_efficient_sorted
    return is_efficient

def _pareto_nd_general(costs):
    """O(n^2) Pareto frontier for N objectives."""
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]: continue
        others = costs[is_efficient]
        dominated = (np.all(others <= costs[i], axis=1) & np.any(others < costs[i], axis=1))
        # Mask self
        efficient_indices = np.where(is_efficient)[0]
        self_pos = np.where(efficient_indices == i)[0][0]
        dominated[self_pos] = False
        if np.any(dominated): is_efficient[i] = False
    return is_efficient

def is_pareto_efficient(costs):
    """Dispatches to the fastest correct Pareto algorithm based on dimension."""
    if costs.shape[0] == 0: return np.array([], dtype=bool)
    return _pareto_2d_fast(costs) if costs.shape[1] == 2 else _pareto_nd_general(costs)

def parse_cell_file(filepath):
    """Extract cell parameters from .cell file."""
    params = {}
    if not os.path.exists(filepath):
        return params
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('-'):
                continue
            parts = line[1:].split(':', 1)
            if len(parts) != 2:
                continue
            k, v = parts
            val = v.strip()
            try:
                val = float(val)
            except ValueError:
                pass
            params[f"CellInput_{k.strip()}"] = val
    return params

OBJECTIVE_PAIRS = [
    ("Cache Hit Latency (ns)", "Cache Area (mm^2)"),
    ("Cache Hit Latency (ns)", "Cache Hit Energy (nJ)"),
    ("Cache Hit Latency (ns)", "Cache Leakage Power (mW)"),
]

def _load_simulation_csv(results_dir, csv_file, mem_type):
    """Load PPA data and attach variant parameters."""
    path = os.path.join(results_dir, csv_file)
    try:
        # Detect format (Cache >= 90 cols, RAM < 90)
        first = pd.read_csv(path, header=None, nrows=1, skipinitialspace=True)
        is_cache = len(first.columns) >= 90
        cols = [1,2,6,10] if is_cache else [24,32,35,38]
        df = pd.read_csv(path, header=None, usecols=cols, skipinitialspace=True)
        if not is_cache: df.iloc[:, 2] /= 1000.0 # pJ -> nJ
        
        df.columns = ["Cache Area (mm^2)", "Cache Hit Latency (ns)", "Cache Hit Energy (nJ)", "Cache Leakage Power (mW)"]
        
        p = csv_file.replace('.csv', '').split('_')
        df['memory_technology'], df['variant_id'], df['capacity_mb'] = mem_type, int(p[2]), float(p[4])/1024.0
        
        for k, v in parse_cell_file(f"synthetic_cells/{mem_type}/synthetic_variant_{p[2]}.cell").items():
            df[k] = v

        # Local Pareto filter to save memory
        ppa = df[["Cache Hit Latency (ns)", "Cache Area (mm^2)"]].dropna()
        return df.loc[ppa.index[is_pareto_efficient(ppa.values)]] if not ppa.empty else df
    except Exception:
        return None

def process_results(mem_types, only_full=False):
    """Orchestrate the Pareto extraction and universal merge.
    Outputs written:
      pareto/{tech}/{tech}_cap_{X}_pareto.csv     one file per capacity point
      pareto/{tech}/{tech}_pareto.csv  union across all capacities
      pareto/pareto.csv               union across all technologies
    """
    for mt in mem_types:
        r_dir, o_dir = f"exploration_results/{mt}", f"pareto/{mt}"
        if not os.path.exists(r_dir): continue
        
        files = [f for f in os.listdir(r_dir) if f.endswith('.csv')]
        if not files: continue

        print(f"Processing {mt} ({len(files)} files)...")
        frames = [_load_simulation_csv(r_dir, f, mt) for f in files]
        full = pd.concat([f for f in frames if f is not None], ignore_index=True).dropna(axis=1, how='all')
        
        os.makedirs(o_dir, exist_ok=True)
        full.to_csv(os.path.join(o_dir, f"{mt}_full_data.csv"), index=False)
        if only_full: continue

        tech_sets = []
        for cap in sorted(full['capacity_mb'].unique()):
            c_df = full[full['capacity_mb'] == cap]
            p_rows = [c_df[is_pareto_efficient(c_df[[x,y]].values)] for x,y in OBJECTIVE_PAIRS if x in c_df and y in c_df]
            if not p_rows: continue
            cap_p = pd.concat(p_rows).drop_duplicates()
            cap_p.to_csv(os.path.join(o_dir, f"{mt}_cap_{str(cap).replace('.','_')}_pareto.csv"), index=False)
            tech_sets.append(cap_p)

        if tech_sets:
            pd.concat(tech_sets).drop_duplicates().to_csv(os.path.join(o_dir, f"{mt}_pareto.csv"), index=False)

    # Universal Merge
    all_p = []
    for t in ["SRAM", "RRAM", "eDRAM"]:
        tp = os.path.join("pareto", t, f"{t}_pareto.csv")
        if os.path.exists(tp): all_p.append(pd.read_csv(tp))
            
    if all_p:
        os.makedirs('pareto', exist_ok=True)
        pd.concat(all_p, ignore_index=True).drop_duplicates().to_csv("pareto/pareto.csv", index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--type", default="ALL")
    p.add_argument("--only-full", action="store_true")
    args = p.parse_args()
    process_results(["SRAM", "RRAM", "eDRAM"] if args.type == "ALL" else [args.type], args.only_full)

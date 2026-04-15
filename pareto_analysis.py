import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def _pareto_2d_fast(costs):
    """O(n log n) Pareto frontier for exactly 2 objectives (lower is better).

    Algorithm: sort by obj0 ascending, sweep tracking the running minimum of
    obj1. A point is on the frontier iff its obj1 strictly improves the best
    seen so far. Handles ties in obj0 correctly via lexsort.
    """
    n = costs.shape[0]
    if n == 0:
        return np.array([], dtype=bool)

    # Sort by obj0 ascending; break ties by obj1 ascending
    sort_idx = np.lexsort((costs[:, 1], costs[:, 0]))
    sorted_costs = costs[sort_idx]

    is_efficient_sorted = np.zeros(n, dtype=bool)
    min_obj1_so_far = np.inf

    for i in range(n):
        if sorted_costs[i, 1] < min_obj1_so_far:
            is_efficient_sorted[i] = True
            min_obj1_so_far = sorted_costs[i, 1]

    # Map back to original row order
    is_efficient = np.zeros(n, dtype=bool)
    is_efficient[sort_idx] = is_efficient_sorted
    return is_efficient


def _pareto_nd_general(costs):
    """O(n²) Pareto frontier for N objectives (lower is better).

    Used as a fallback when more than 2 objectives are requested (e.g.,
    Latency vs. Energy vs. Area). 
    Slow - only call on pre-filtered data or small datasets.
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Point i is dominated if any other efficient point is <= in all dims
        # and strictly < in at least one
        others = costs[is_efficient]
        dominated_by_others = (
            np.all(others <= costs[i], axis=1) &
            np.any(others < costs[i], axis=1)
        )
        # Exclude self-comparison (index of i within the efficient subset)
        efficient_indices = np.where(is_efficient)[0]
        self_pos = np.where(efficient_indices == i)[0][0]
        dominated_by_others[self_pos] = False
        if np.any(dominated_by_others):
            is_efficient[i] = False
    return is_efficient


def is_pareto_efficient(costs):
    """Fastest correct Pareto algorithm.

    - 2 objectives  → O(n log n) sort-and-sweep  (_pareto_2d_fast)
    - 3+ objectives → O(n²) general N-dim check  (_pareto_nd_general)

    To add Energy as a 3rd objective, just pass a (n, 3) costs array;
    the dispatch happens automatically with no changes needed at call sites.

    Args:
        costs: np.ndarray of shape (n, k), k = number of objectives.
               All objectives are minimized (lower is better).
    Returns:
        Boolean mask of length n; True = point is on the Pareto frontier.
    """
    if costs.shape[0] == 0:
        return np.array([], dtype=bool)

    n_objectives = costs.shape[1]
    if n_objectives == 2:
        return _pareto_2d_fast(costs)
    else:
        return _pareto_nd_general(costs)

def parse_cell_file(filepath):
    """Parse the input configuration used for this variant."""
    params = {}
    if not os.path.exists(filepath):
        return params
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('-'):
                try:
                    key_part, val_part = line[1:].split(':', 1)
                    val = val_part.strip()
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    params[f"CellInput_{key_part.strip()}"] = val
                except ValueError:
                    pass
    return params

# Objective pairs used for per-capacity Pareto filtering.
# Using multiple 2D pairs (rather than one 4D filter) keeps the fast O(n log n)
# algorithm and captures diverse tradeoff structure across the Pareto surface.
OBJECTIVE_PAIRS = [
    ("Cache Hit Latency (ns)", "Cache Area (mm^2)"),
    ("Cache Hit Latency (ns)", "Cache Hit Energy (nJ)"),
    ("Cache Hit Latency (ns)", "Cache Leakage Power (mW)"),
]


def _load_simulation_csv(results_dir, csv_file, mem_type):
    """Load one DESTINY output CSV and return a DataFrame with PPA + cell parameters.

    DESTINY produces two different CSV layouts depending on the design target:
      - Cache format (>=90 cols): used by SRAM and eDRAM simulations.
      - RAM   format ( <90 cols): used by RRAM simulations; energy is in pJ.

    Returns None on any error (file is silently skipped by the caller).
    """
    filepath = os.path.join(results_dir, csv_file)
    try:
        first_row = pd.read_csv(filepath, header=None, nrows=1, skipinitialspace=True)
        num_cols  = len(first_row.columns)

        if num_cols >= 90:  # Cache format: col 1=Area, 2=HitLatency, 6=HitEnergy, 10=Leakage
            df = pd.read_csv(filepath, header=None, usecols=[1, 2, 6, 10], skipinitialspace=True)
        else:               # RAM format:   col 24=Area, 32=Latency, 35=Energy(pJ), 38=Leakage
            if 38 >= num_cols:
                raise ValueError(f"RAM format: only {num_cols} columns, expected >= 39")
            df = pd.read_csv(filepath, header=None, usecols=[24, 32, 35, 38], skipinitialspace=True)
            df.iloc[:, 2] /= 1000.0  # energy: pJ -> nJ

        df.columns = ["Cache Area (mm^2)", "Cache Hit Latency (ns)",
                      "Cache Hit Energy (nJ)", "Cache Leakage Power (mW)"]

        # Parse variant number and capacity from filename: e.g. variant_857_cap_4096.csv
        parts       = csv_file.replace('.csv', '').split('_')
        variant_num = int(parts[2])
        capacity_kb = float(parts[4])

        df['memory_technology'] = mem_type
        df['variant_id']        = variant_num
        df['capacity_mb']       = capacity_kb / 1024.0

        cell_file = f"synthetic_cells/{mem_type}/synthetic_variant_{variant_num}.cell"
        for key, val in parse_cell_file(cell_file).items():
            df[key] = val

        # Pre-filter to the local Pareto frontier before returning.
        # Any point dominated within its own file is also globally dominated,
        # so this is safe and drastically reduces peak RAM on large RRAM files.
        ppa = df[["Cache Hit Latency (ns)", "Cache Area (mm^2)"]].dropna()
        if len(ppa) > 0:
            df = df.loc[ppa.index[is_pareto_efficient(ppa.values)]]

        return df

    except Exception as e:
        print(f"  Warning: could not load {csv_file}: {e}")
        return None


def process_results(mem_types):
    """For each technology: load simulations, compute per-capacity Pareto sets, save CSVs.

    Outputs written:
      pareto/{tech}/{tech}_cap_{X}_pareto.csv     one file per capacity point
      pareto/{tech}/{tech}_training_dataset_pareto.csv  union across all capacities
      pareto/training_dataset_pareto.csv               union across all technologies
    """
    all_tech_pareto = []  # one deduplicated DataFrame per technology

    for mem_type in mem_types:
        results_dir = f"exploration_results/{mem_type}"
        output_dir  = f"pareto/{mem_type}"
        tech_csv    = os.path.join(output_dir, f"{mem_type}_training_dataset_pareto.csv")

        csv_files = ([f for f in os.listdir(results_dir) if f.endswith('.csv')]
                     if os.path.exists(results_dir) else [])
        if not csv_files:
            print(f"No CSVs found in {results_dir}. Skipping {mem_type}.")
            continue

        print(f"Processing {len(csv_files)} files for {mem_type}...")

        # Load every simulation file (locally pre-filtered inside _load_simulation_csv)
        frames     = [_load_simulation_csv(results_dir, f, mem_type) for f in csv_files]
        full_space = pd.concat(
            [f for f in frames if f is not None], ignore_index=True
        ).dropna(axis=1, how='all')

        os.makedirs(output_dir, exist_ok=True)

        # Per-capacity Pareto: for each cache size, keep the best designs
        # across all objective pairs, then write a per-capacity CSV.
        tech_rows = []
        for cap in sorted(full_space['capacity_mb'].unique()):
            cap_df   = full_space[full_space['capacity_mb'] == cap]
            cap_rows = [
                cap_df[is_pareto_efficient(cap_df[[x, y]].values)]
                for x, y in OBJECTIVE_PAIRS
                if x in cap_df.columns and y in cap_df.columns
            ]
            if not cap_rows:
                continue
            cap_pareto = pd.concat(cap_rows).drop_duplicates()
            cap_label  = str(cap).replace('.', '_')
            cap_pareto.to_csv(
                os.path.join(output_dir, f"{mem_type}_cap_{cap_label}_pareto.csv"),
                index=False
            )
            tech_rows.append(cap_pareto)

        if not tech_rows:
            print(f"[{mem_type}] No Pareto-optimal designs found.")
            continue

        # Tech-level summary = union of all per-capacity Pareto sets
        tech_pareto = pd.concat(tech_rows).drop_duplicates()
        tech_pareto.to_csv(tech_csv, index=False)
        all_tech_pareto.append(tech_pareto)
        print(f"[{mem_type}] {len(tech_pareto)} designs saved "
              f"(union across {len(tech_rows)} capacity points) -> {tech_csv}")

    # Universal dataset = union across all technologies
    if all_tech_pareto:
        universal = pd.concat(all_tech_pareto).drop_duplicates()
        os.makedirs('pareto', exist_ok=True)
        universal.to_csv("pareto/training_dataset_pareto.csv", index=False)
        print(f"\nUniversal dataset: {len(universal)} designs "
              f"-> pareto/training_dataset_pareto.csv")
    else:
        print("No valid results found to process.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hardware sweep results.")
    parser.add_argument("--type", type=str, default="ALL", help="Memory type (SRAM, RRAM, eDRAM, ALL)")
    args = parser.parse_args()
    
    mem_types = ["SRAM", "RRAM", "eDRAM"] if args.type == "ALL" else [args.type]
    process_results(mem_types)

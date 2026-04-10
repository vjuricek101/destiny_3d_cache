import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def is_pareto_efficient(costs):
    """Find the pareto-efficient points (lower is better for all costs)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  
            is_efficient[i] = True  
            is_efficient[is_efficient] &= ~np.all(costs[is_efficient] >= c, axis=1) | (np.arange(len(is_efficient))[is_efficient] == i)
    return is_efficient

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

def process_results(mem_types):
    all_pareto_sets = []
    
    for mem_type in mem_types:
        results_dir = f"exploration_results/{mem_type}"
        output_dir = f"pareto/{mem_type}"
        dataset_output = os.path.join(output_dir, f"{mem_type}_training_dataset_pareto.csv")

        all_dataframes = []
        if not os.path.exists(results_dir):
            print(f"Directory {results_dir} not found for {mem_type}. Skipping...")
            continue

        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        if not csv_files:
            continue

        print(f"Processing {len(csv_files)} hardware sweep results for {mem_type}...")
        
        column_names = [
            "Access Mode", "Cache Area (mm^2)", "Cache Hit Latency (ns)", "Cache Miss Latency (ns)",
            "Cache Write Latency (ns)", "Cache Refresh Latency (ns)", "Cache Hit Energy (nJ)",
            "Cache Miss Energy (nJ)", "Cache Write Energy (nJ)", "Cache Refresh Energy (nJ)",
            "Cache Leakage Power (mW)", "Cache Refresh Power (W)"
        ]
        column_names += [f"ArchParam_{i}" for i in range(len(column_names), 100)]

        for csv_file in csv_files:
            filepath = os.path.join(results_dir, csv_file)
            try:
                df = pd.read_csv(filepath, header=None, skipinitialspace=True)
                df.columns = column_names[:len(df.columns)]
                
                parts = csv_file.replace('.csv', '').split('_')
                variant_num = int(parts[2])
                capacity = float(parts[4])
                
                # Fetch Cell parameters used for this run
                cell_file = f"synthetic_cells/{mem_type}/synthetic_variant_{variant_num}.cell"
                cell_params = parse_cell_file(cell_file)
                
                df['memory_technology'] = mem_type
                df['variant_id'] = variant_num
                df['capacity_mb'] = capacity
                
                for key, val in cell_params.items():
                    df[key] = val
                    
                all_dataframes.append(df)
            except Exception as e:
                pass

        if not all_dataframes:
            continue
            
        full_space = pd.concat(all_dataframes, ignore_index=True)
        full_space = full_space.dropna(axis=1, how='all')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the per-technology aggregated dataset
        full_space.to_csv(dataset_output, index=False)
        print(f"[{mem_type}] Saved aggregated dataset.")

        objective_pairs = [
            ("Cache Hit Latency (ns)", "Cache Leakage Power (mW)"),
            ("Cache Hit Latency (ns)", "Cache Hit Energy (nJ)"),
            ("Cache Area (mm^2)", "Cache Hit Latency (ns)")
        ]

        capacities = sorted(full_space['capacity_mb'].unique())
        
        for cap in capacities:
            cap_df = full_space[full_space['capacity_mb'] == cap].copy()
            for x_metric, y_metric in objective_pairs:
                if x_metric in cap_df.columns and y_metric in cap_df.columns:
                    costs = cap_df[[x_metric, y_metric]].values
                    pareto_mask = is_pareto_efficient(costs)
                    all_pareto_sets.append(cap_df[pareto_mask].copy())

    if all_pareto_sets:
        final_optimal_dataset = pd.concat(all_pareto_sets).drop_duplicates()
        if not os.path.exists('pareto'):
            os.makedirs('pareto')
            
        universal_dataset = "pareto/training_dataset_pareto.csv"
        final_optimal_dataset.dropna(axis=1, how='all', inplace=True)
        final_optimal_dataset.to_csv(universal_dataset, index=False)
        print(f"\nFinal training dataset saved: {universal_dataset}")
        print(f"Total optimal designs: {len(final_optimal_dataset)}")
    else:
        print("No valid results found to process.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hardware sweep results.")
    parser.add_argument("--type", type=str, default="ALL", help="Memory type (SRAM, RRAM, eDRAM, ALL)")
    args = parser.parse_args()
    
    mem_types = ["SRAM", "RRAM", "eDRAM"] if args.type == "ALL" else [args.type]
    process_results(mem_types)

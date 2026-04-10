import os
import re
import subprocess
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Map types to their configuration templates ---
CFG_TEMPLATES = {
    "SRAM": "config/sample_SRAM_2layer.cfg",
    "RRAM": "config/sample_2DReRAM.cfg",
    "eDRAM": "config/sample_2D_eDRAM.cfg"
}

# Capacity Sweep: 2KB to 32MB
CAPACITY_SWEEP_KB = [ 2**i for i in range(1, 16)] 

def setup_dirs(mem_type):
    temp_dir = f"temp_configs/{mem_type}"
    results_dir = f"exploration_results/{mem_type}"
    
    for d in [temp_dir, results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    return temp_dir, results_dir

def run_single_simulation(args):
    """Worker function for running a single DESTINY configuration."""
    cap_kb, cell_path, variant_name, base_cfg_content, temp_dir, results_dir = args
    
    # Mutate capacity and point to the synthetic cell
    new_cfg = re.sub(r"-Capacity\s*\(MB\):.*", f"-Capacity (KB): {cap_kb}", base_cfg_content)
    new_cfg = re.sub(r"-Capacity\s*\(KB\):.*", f"-Capacity (KB): {cap_kb}", new_cfg)
    abs_cell_path = os.path.abspath(cell_path)
    new_cfg = re.sub(r"-MemoryCellInputFile:.*", f"-MemoryCellInputFile: {abs_cell_path}", new_cfg)
    
    # Force Full optimization for Pareto extraction
    new_cfg = re.sub(r"^[/-]*OptimizationTarget:.*", "", new_cfg, flags=re.MULTILINE)
    new_cfg += "\n-OptimizationTarget: Full\n"
    
    cfg_filename = f"{variant_name}_cap_{cap_kb}.cfg"
    cfg_filepath = os.path.join(temp_dir, cfg_filename)
    
    with open(cfg_filepath, 'w') as f:
        f.write(new_cfg)
    
    try:
        # Run destiny from the root
        subprocess.run(["./destiny", cfg_filepath], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Relocate the CSV output
        expected_csv = os.path.join(temp_dir, f"{variant_name}_cap_{cap_kb}.csv")
        if os.path.exists(expected_csv):
            final_csv = os.path.join(results_dir, f"{variant_name}_cap_{cap_kb}.csv")
            shutil.move(expected_csv, final_csv)
            return True
        return False
    except:
        return False

def generate_and_run(mem_type):
    # Sanity check for binary
    if not os.path.exists("./destiny"):
        print("Error: 'destiny' binary not found. Run 'make' first.")
        return

    temp_dir, results_dir = setup_dirs(mem_type)
    base_cfg_file = CFG_TEMPLATES.get(mem_type)
    if not base_cfg_file:
        print(f"Error: Unsupported type '{mem_type}'")
        return

    with open(base_cfg_file, 'r') as f:
        base_cfg_content = f.read()

    cell_dir = f"synthetic_cells/{mem_type}"
    if not os.path.exists(cell_dir):
        print(f"No synthetic cells found for {mem_type}. Run generate_cells.py first.")
        return

    cells = [f for f in os.listdir(cell_dir) if f.endswith('.cell')]
    
    # Generate the argument list for the thread pool
    simulation_args = []
    for cell_file in cells:
        cell_path = os.path.join(cell_dir, cell_file)
        variant_name = cell_file.split('.')[0]
        for cap_kb in CAPACITY_SWEEP_KB:
            simulation_args.append(
                (cap_kb, cell_path, variant_name, base_cfg_content, temp_dir, results_dir)
            )
            
    total_runs = len(simulation_args)
    print(f"Launching {total_runs} parallel DESTINY simulations for {mem_type}...")
    
    run_count = 0
    # Use max_workers based on available CPU cores. Default is os.cpu_count() + 4 by default in Python 3.8+
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_simulation, arg): arg for arg in simulation_args}
        for future in as_completed(futures):
            run_count += 1
            if run_count % 50 == 0 or run_count == total_runs:
                print(f"[{run_count}/{total_runs}] simulations completed.")
                
    print(f"Finished {mem_type} exploration sweep.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full design-space exploration.")
    parser.add_argument("--type", type=str, default="ALL", help="Memory type (SRAM, RRAM, eDRAM, ALL)")
    args = parser.parse_args()
    
    if args.type == "ALL":
        for mem in CFG_TEMPLATES.keys():
            generate_and_run(mem)
    else:
        generate_and_run(args.type)
        
    print("\nStep 2 Complete: Architecture Exploration Sweep Done.")

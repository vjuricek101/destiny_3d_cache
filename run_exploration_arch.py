#!/usr/bin/env python3
import os
import re
import subprocess
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

# ── Physics Tables ─────────────────────────────────────────────────────────────

# Global Temperature points for independent thermal sweep (K)
TEMPERATURE_SWEEP: List[int] = [300, 350, 400, 450]

BITLINE_LEAKAGE_TOLERANCE = 1
MAX_WORKERS: int = os.cpu_count() or 64

# ── Architectural Design Space ────────────────────────────────────────────────

# Granular capacity sweep: 2KB to 32MB
CAPACITY_SWEEP_KB: List[int] = [2**i for i in range(1, 16)] 

# Expanded architectural candidates (Power-of-2 only for DESTINY compatibility)
WORD_WIDTHS: List[int]     = [64, 128, 256, 512, 1024, 2048]
ASSOCIATIVITIES: List[int] = [1, 2, 4, 8, 16, 32, 64]
STACK_COUNTS: List[int]    = [1, 2, 4, 8, 16]

# ── Config Templates & Roadmaps ────────────────────────────────────────────────
CFG_TEMPLATES: Dict[str, str] = {
    "SRAM":  "config/sample_SRAM_2layer.cfg",
    "RRAM":  "config/sample_2DReRAM.cfg",
    "eDRAM": "config/sample_2D_eDRAM.cfg",
}

ROADMAPS: Dict[str, List[str]] = {
    "SRAM":  ["HP", "LOP", "LSTP"],
    "RRAM":  ["HP"],
    "eDRAM": ["EDRAM"],
}

# ── Utilities ──────────────────────────────────────────────────────────────────

def parse_cell_params(filepath: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'): continue
            if line.startswith('-'):
                parts = line[1:].split(':', 1)
                if len(parts) == 2: params[parts[0].strip()] = parts[1].strip()
    return params

def extract_process_node(cell_filename: str) -> Optional[int]:
    m = re.search(r'_n(\d+)\.cell$', cell_filename)
    return int(m.group(1)) if m else None

def setup_dirs(mem_type: str) -> Tuple[str, str]:
    temp_dir    = f"/dev/shm/vjuricek_destiny_tmp/{mem_type}_arch"
    results_dir = f"exploration_results/{mem_type}_arch"
    for d in [temp_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    return temp_dir, results_dir

# ── Worker Function ────────────────────────────────────────────────────────────

def run_single_simulation(args: tuple) -> bool:
    (cap_kb, cell_path, variant_name, roadmap, base_cfg_content,
     temp_dir, results_dir, cfg_overrides, cfg_suffix) = args

    new_cfg = re.sub(r"-Capacity\s*\(MB\):.*", f"-Capacity (KB): {cap_kb}", base_cfg_content)
    new_cfg = re.sub(r"-Capacity\s*\(KB\):.*", f"-Capacity (KB): {cap_kb}", new_cfg)
    
    abs_cell = os.path.abspath(cell_path)
    new_cfg  = re.sub(r"-MemoryCellInputFile:.*", f"-MemoryCellInputFile: {abs_cell}", new_cfg)
    new_cfg  = re.sub(r"^[/-]*OptimizationTarget:.*", "", new_cfg, flags=re.MULTILINE)
    new_cfg += "\n-OptimizationTarget: Full\n"
    new_cfg = re.sub(r"^[/-]*DeviceRoadmap:.*", f"-DeviceRoadmap: {roadmap}", new_cfg, flags=re.MULTILINE)

    for param, value in cfg_overrides.items():
        pattern = rf"^[/-]*{re.escape(param)}:.*"
        replacement = f"-{param}: {value}"
        if re.search(pattern, new_cfg, flags=re.MULTILINE):
            new_cfg = re.sub(pattern, replacement, new_cfg, flags=re.MULTILINE)
        else:
            new_cfg += f"\n-{param}: {value}\n"

    cfg_filename = f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}.cfg"
    cfg_filepath = os.path.join(temp_dir, cfg_filename)

    with open(cfg_filepath, 'w') as f: f.write(new_cfg)

    try:
        res = subprocess.run(["./destiny", cfg_filepath], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0: return False
        expected_csv = cfg_filepath.replace(".cfg", ".csv")
        if os.path.exists(expected_csv):
            final_csv = os.path.join(results_dir, cfg_filename.replace(".cfg", ".csv"))
            shutil.move(expected_csv, final_csv)
            return True
        return False
    finally:
        if os.path.exists(cfg_filepath): os.remove(cfg_filepath)

# ── Orchestrator ───────────────────────────────────────────────────────────────

def collect_simulations(mem_type: str) -> List[tuple]:
    temp_dir, results_dir = setup_dirs(mem_type)
    base_cfg_file = CFG_TEMPLATES.get(mem_type)
    with open(base_cfg_file, 'r') as f: base_cfg_content = f.read()

    cell_dir = f"synthetic_cells/{mem_type}_arch"
    if not os.path.exists(cell_dir):
        print(f"No architectural cells found for {mem_type}. Run generate_cells_arch.py first.")
        return []

    cells = sorted(f for f in os.listdir(cell_dir) if f.endswith('.cell'))
    roadmaps = ROADMAPS.get(mem_type, ["HP"])
    simulation_args = []

    for cell_file in cells:
        cell_path = os.path.join(cell_dir, cell_file)
        variant_name = cell_file.split('.')[0]
        process_node = int(extract_process_node(cell_file))
        cell_params  = parse_cell_params(cell_path)

        for cap_kb in CAPACITY_SWEEP_KB:
            for stacked in STACK_COUNTS:
                for temperature in TEMPERATURE_SWEEP:
                    cfg_base = {
                        "Temperature (K)": temperature,
                        "StackedDieCount":  stacked,
                        "ProcessNode":      process_node,
                    }

                    if mem_type in ("SRAM", "eDRAM"):
                        for assoc in ASSOCIATIVITIES:
                            for ww_bits in WORD_WIDTHS:
                                overrides = dict(cfg_base)
                                overrides["WordWidth (bit)"]                = ww_bits
                                overrides["Associativity (for cache only)"] = assoc
                                if mem_type == "eDRAM":
                                    overrides["RetentionTime (us)"] = int(float(cell_params.get("RetentionTime (us)", 20)))
                                
                                suffix = f"_ww{ww_bits}_a{assoc}_s{stacked}_t{temperature}"
                                for roadmap in roadmaps:
                                    simulation_args.append((cap_kb, cell_path, variant_name, roadmap, base_cfg_content, temp_dir, results_dir, overrides, suffix))

                    elif mem_type == "RRAM":
                        sensing_options = [True, False] # Always sweep both for arch sweep
                        for ww_bits in WORD_WIDTHS:
                            for sensing in sensing_options:
                                overrides = dict(cfg_base)
                                overrides["WordWidth (bit)"] = ww_bits
                                overrides["InternalSensing"] = "true" if sensing else "false"
                                suffix = f"_ww{ww_bits}_sens{'T' if sensing else 'F'}_s{stacked}_t{temperature}"
                                for roadmap in roadmaps:
                                    simulation_args.append((cap_kb, cell_path, variant_name, roadmap, base_cfg_content, temp_dir, results_dir, overrides, suffix))

    return simulation_args

def execute_simulations(simulation_args: List[tuple], label: str):
    if not simulation_args: return
    print(f"\nLaunching {len(simulation_args)} architectural sweep simulations for {label}...")
    run_count = success_count = 0
    total_runs = len(simulation_args)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_simulation, arg): arg for arg in simulation_args}
        for future in as_completed(futures):
            run_count += 1
            if future.result(): success_count += 1
            if run_count % 500 == 0 or run_count == total_runs:
                print(f"  [{run_count}/{total_runs}] completed ({success_count} outputs).")

def main():
    parser = argparse.ArgumentParser(description="Dense Cartesian architectural sweep for DESTINY.")
    parser.add_argument("--type", type=str, default="ALL")
    args = parser.parse_args()

    types = ["SRAM", "RRAM", "eDRAM"] if args.type.upper() == "ALL" else [args.type]
    for t in types:
        sim_args = collect_simulations(t)
        execute_simulations(sim_args, t + "_arch")

    print("\nArchitectural Sweep Done.")

if __name__ == "__main__":
    main()

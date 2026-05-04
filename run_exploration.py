#!/usr/bin/env python3
import os
import re
import subprocess
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from destiny_utils import KEEP_COLS, parse_cell_params, extract_process_node, setup_dirs

# ── Configuration & Sweep Parameters ──────────────────────────────────────────
MAX_WORKERS: int = 128
BITLINE_LEAKAGE_TOLERANCE = 1

CAPACITY_SWEEP_KB: List[int] = [2**i for i in range(1, 16)]   # 2 KB – 32 MB
STACK_COUNTS: List[int]       = [1, 2, 4]
ASSOCIATIVITIES: List[int]    = [4]

# Temperature Map: MemoryType -> StackCount -> List of Temperature (K)
TEMPERATURE_MAP: Dict[str, Dict[int, List[int]]] = {
    "SRAM":  {1: [300], 2: [363], 4: [380]},
    "eDRAM": {1: [350], 2: [363], 4: [380]},
    "RRAM":  {1: [313], 2: [333], 4: [358]},
}

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

# ── Capacity Buckets ──────────────────────────────────────────────────────────

def capacity_bucket_word_widths(cap_kb: int) -> List[int]:
    """WordWidth options based on capacity."""
    if cap_kb < 1024:        return [64, 128]
    if cap_kb <= 16 * 1024:  return [128, 256, 512]
    return [512, 1024]

# ── RRAM Specific Helpers ─────────────────────────────────────────────────────

def rram_internal_sensing_options(cell: Dict[str, str]) -> List[bool]:
    try:
        ron   = float(cell["ResistanceOnAtSetVoltage (ohm)"])
        roff  = float(cell["ResistanceOffAtSetVoltage (ohm)"])
        ratio = roff / ron
    except (KeyError, ZeroDivisionError, ValueError):
        return [True]
    return [True] if ratio < 100 else ([False] if ratio > 1000 else [True, False])

def filter1_crossbar_ok(cell: Dict[str, str], cap_kb: int, ww_bits: int) -> bool:
    access = cell.get("AccessType", "none").strip().lower()
    if access != "none": return True
    ron_half  = float(cell.get("ResistanceOnAtHalfResetVoltage (ohm)", 1e3))
    roff_read = float(cell.get("ResistanceOffAtReadVoltage (ohm)", 1e6))
    max_rows  = roff_read / (2.0 * ron_half * BITLINE_LEAKAGE_TOLERANCE)
    ceiling_kb = (max_rows * ww_bits * 0.1) / (8 * 1024)
    return cap_kb < ceiling_kb

def filter2_high_ron_ok(cell: Dict[str, str], cap_kb: int) -> bool:
    ron = float(cell.get("ResistanceOnAtSetVoltage (ohm)", 0))
    return not (ron > 1e6 and cap_kb > 8 * 1024)

# ── Worker ────────────────────────────────────────────────────────────────────

def run_single_simulation(args: tuple) -> bool:
    (cap_kb, cell_path, variant_name, roadmap, base_cfg_content,
     temp_dir, results_dir, cfg_overrides, cfg_suffix, mem_type) = args

    final_csv = os.path.join(results_dir, f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}.csv")
    if os.path.exists(final_csv): return True

    # Build config
    new_cfg = re.sub(r"-Capacity\s*\(MB\):.*", f"-Capacity (KB): {cap_kb}", base_cfg_content)
    new_cfg = re.sub(r"-Capacity\s*\(KB\):.*", f"-Capacity (KB): {cap_kb}", new_cfg)
    new_cfg = re.sub(r"-MemoryCellInputFile:.*", f"-MemoryCellInputFile: {os.path.abspath(cell_path)}", new_cfg)
    
    # Force full exploration with pruning
    for param in ["OptimizationTarget", "EnablePruning"]:
        new_cfg = re.sub(rf"^[/-]*{param}:.*", "", new_cfg, flags=re.MULTILINE)
    new_cfg += "\n-OptimizationTarget: Full\n-EnablePruning: Yes\n"

    new_cfg = re.sub(r"^[/-]*DeviceRoadmap:.*", f"-DeviceRoadmap: {roadmap}", new_cfg, flags=re.MULTILINE)
    
    for param, value in cfg_overrides.items():
        pattern = rf"^[/-]*{re.escape(param)}:.*"
        replacement = f"-{param}: {value}"
        if re.search(pattern, new_cfg, flags=re.MULTILINE):
            new_cfg = re.sub(pattern, replacement, new_cfg, flags=re.MULTILINE)
        else:
            new_cfg += f"\n-{param}: {value}\n"

    cfg_filepath = os.path.join(temp_dir, f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}.cfg")
    with open(cfg_filepath, 'w') as f: f.write(new_cfg)

    try:
        res = subprocess.run(["./destiny", cfg_filepath], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0: return False

        expected_csv = cfg_filepath.replace(".cfg", ".csv")
        if not os.path.exists(expected_csv): return False

        df = pd.read_csv(expected_csv)
        if df.empty: return False
        df.insert(0, "variant_name", variant_name)
        df.insert(1, "technology", mem_type)
        
        cell_params = parse_cell_params(cell_path)
        for k, v in cell_params.items():
            try: df[f"CellInput_{k}"] = float(v)
            except ValueError: df[f"CellInput_{k}"] = v
                
        base_cols = [c for c in KEEP_COLS if c in df.columns]
        cell_cols = [c for c in df.columns if c.startswith("CellInput_")]
        df[base_cols + cell_cols].to_csv(final_csv, index=False)
        os.remove(expected_csv)
        return True
    finally:
        if os.path.exists(cfg_filepath): os.remove(cfg_filepath)

# ── Simulation Argument Builder ───────────────────────────────────────────────

def build_simulation_args(
    mem_type: str, cells: List[str], cell_dir: str, base_cfg_content: str, temp_dir: str, results_dir: str,
) -> Tuple[List[tuple], Dict[str, int]]:
    simulation_args: List[tuple] = []
    rejected: Dict[str, int] = {"L5_F1_crossbar": 0, "L5_F2_high_ron": 0}
    roadmaps = ROADMAPS.get(mem_type, ["HP"])

    for cell_file in cells:
        cell_path    = os.path.join(cell_dir, cell_file)
        variant_name = cell_file.split('.')[0]
        process_node = int(extract_process_node(cell_file))
        cell_params  = parse_cell_params(cell_path)

        for cap_kb in CAPACITY_SWEEP_KB:
            for stacked in STACK_COUNTS:
                for temperature in TEMPERATURE_MAP[mem_type].get(stacked, [350]):
                    cfg_base = {
                        "Temperature (K)": temperature,
                        "StackedDieCount":  stacked,
                        "ProcessNode":      process_node,
                    }

                    if mem_type in ("SRAM", "eDRAM"):
                        for assoc in ASSOCIATIVITIES:
                            for ww_bits in capacity_bucket_word_widths(cap_kb):
                                overrides = dict(cfg_base)
                                overrides["WordWidth (bit)"]                = ww_bits
                                overrides["Associativity (for cache only)"] = assoc
                                if mem_type == "eDRAM":
                                    overrides["RetentionTime (us)"] = int(float(cell_params.get("RetentionTime (us)", 20.0)))

                                suffix = f"_ww{ww_bits}_a{assoc}_s{stacked}_t{temperature}"
                                for roadmap in roadmaps:
                                    simulation_args.append((
                                        cap_kb, cell_path, variant_name, roadmap, base_cfg_content,
                                        temp_dir, results_dir, overrides, suffix, mem_type
                                    ))

                    elif mem_type == "RRAM":
                        ww_options      = capacity_bucket_word_widths(cap_kb)
                        sensing_options = rram_internal_sensing_options(cell_params)
                        for ww_bits in ww_options:
                            if not filter1_crossbar_ok(cell_params, cap_kb, ww_bits):
                                rejected["L5_F1_crossbar"] += len(sensing_options)
                                continue
                            if not filter2_high_ron_ok(cell_params, cap_kb):
                                rejected["L5_F2_high_ron"] += len(sensing_options)
                                continue
                            for sensing in sensing_options:
                                for assoc in ASSOCIATIVITIES:
                                    overrides = dict(cfg_base)
                                    overrides["WordWidth (bit)"]                = ww_bits
                                    overrides["InternalSensing"]                = "true" if sensing else "false"
                                    overrides["Associativity (for cache only)"] = assoc
                                    suffix = f"_ww{ww_bits}_sens{'T' if sensing else 'F'}_a{assoc}_s{stacked}_t{temperature}"
                                    for roadmap in roadmaps:
                                        simulation_args.append((
                                            cap_kb, cell_path, variant_name, roadmap, base_cfg_content,
                                            temp_dir, results_dir, overrides, suffix, mem_type
                                        ))

    return simulation_args, rejected

# ── Main Orchestrator ─────────────────────────────────────────────────────────

def collect_simulations(mem_type: str) -> List[tuple]:
    temp_dir, results_dir = setup_dirs(mem_type, is_arch=False)
    base_cfg_file = CFG_TEMPLATES.get(mem_type)
    if not base_cfg_file: return []

    with open(base_cfg_file, 'r') as f: base_cfg_content = f.read()
    cell_dir = f"synthetic_cells/{mem_type}"
    if not os.path.exists(cell_dir): return []

    cells = sorted(f for f in os.listdir(cell_dir) if f.endswith('.cell'))
    sim_args, rejected = build_simulation_args(mem_type, cells, cell_dir, base_cfg_content, temp_dir, results_dir)
    print(f"  {mem_type}: Valid: {len(sim_args)} | Rejected: {sum(rejected.values())}")
    return sim_args

def execute_simulations(simulation_args: List[tuple], label: str):
    if not simulation_args: return
    print(f"Launching {len(simulation_args)} parallel simulations for {label}...")
    run_count = success_count = 0
    total_runs = len(simulation_args)
    failed_designs = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_simulation, arg): arg for arg in simulation_args}
        for future in as_completed(futures):
            run_count += 1
            if future.result(): success_count += 1
            else:
                cap_kb, _, variant_name, roadmap, _, _, _, overrides, _, mem_type = futures[future]
                failed_designs.append({
                    "capacity_kb": cap_kb, "device_roadmap": roadmap, "variant_name": variant_name,
                    "technology": mem_type,
                    "temperature_K": overrides.get("Temperature (K)"), "process_node_nm": overrides.get("ProcessNode"),
                    "data_stacked_die_count": overrides.get("StackedDieCount"), "word_width_bits": overrides.get("WordWidth (bit)"),
                    "associativity": overrides.get("Associativity (for cache only)"), "is_valid": 0
                })
            if run_count % 1000 == 0 or run_count == total_runs:
                print(f"PROGRESS: {run_count}/{total_runs} completed")

    if failed_designs:
        out_dir = os.path.join("pareto", label)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(failed_designs).to_csv(os.path.join(out_dir, f"{label}_failed.csv"), index=False)

    print(f"Success: {success_count}/{total_runs} simulations saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="ALL", help="Memory type: SRAM, RRAM, eDRAM, ALL")
    args = parser.parse_args()


    mem_types = ["SRAM", "RRAM", "eDRAM"] if args.type.upper() == "ALL" else [args.type.upper()]
    all_args = []
    for mt in mem_types:
        all_args.extend(collect_simulations(mt))
    
    if all_args: execute_simulations(all_args, args.type.upper())
    print("\nExploration Sweep Done.")

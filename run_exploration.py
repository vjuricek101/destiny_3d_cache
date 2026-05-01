#!/usr/bin/env python3
import os
import re
import subprocess
import argparse
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from destiny_utils import KEEP_COLS, parse_cell_params, extract_process_node, setup_dirs

# ── Physics Tables ────────────────────────────────────────────────────────────
# Temperature derived from StackedDieCount (Level 4).
TEMPERATURE_FROM_STACK: Dict[str, Dict[int, int]] = {
    "SRAM":  {1: 350, 2: 363, 4: 380},
    "eDRAM": {1: 350, 2: 363, 4: 380},
    "RRAM":  {1: 313, 2: 333, 4: 358},
}

BITLINE_LEAKAGE_TOLERANCE = 1   # from constant.h
MAX_WORKERS: int = 64

CAPACITY_SWEEP_KB: List[int] = [2**i for i in range(1, 16)]   # 2 KB – 32 MB

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
    """RAM-mode WordWidth options from the Capacity bucket (Level 3)."""
    if cap_kb < 1024:        return [64, 128]
    if cap_kb <= 16 * 1024:  return [128, 256, 512]
    return [512, 1024]

def capacity_bucket_associativities(cap_kb: int) -> List[int]:
    """Cache-mode Associativity options from the Capacity bucket (Level 3)."""
    if cap_kb < 512:         return [1, 2, 4]
    if cap_kb <= 4 * 1024:   return [4, 8, 16]
    if cap_kb <= 32 * 1024:  return [8, 16, 32]
    return [16, 32]



# ── InternalSensing (RRAM) ────────────────────────────────────────────────────

def rram_internal_sensing_options(cell: Dict[str, str]) -> List[bool]:
    try:
        ron   = float(cell["ResistanceOnAtSetVoltage (ohm)"])
        roff  = float(cell["ResistanceOffAtSetVoltage (ohm)"])
        ratio = roff / ron
    except (KeyError, ZeroDivisionError, ValueError):
        return [True]
    if ratio < 100:   return [True]
    if ratio > 1000:  return [False]
    return [True, False]

# ── Cross-Level Validity Filters ──────────────────────────────────────────────

def _max_rows_crossbar(cell: Dict[str, str]) -> float:
    ron_half  = float(cell["ResistanceOnAtHalfResetVoltage (ohm)"])
    roff_read = float(cell["ResistanceOffAtReadVoltage (ohm)"])
    return roff_read / (2.0 * ron_half * BITLINE_LEAKAGE_TOLERANCE)

def crossbar_capacity_ceiling_kb(cell: Dict[str, str], ww_bits: int) -> float:
    return (_max_rows_crossbar(cell) * ww_bits * 0.1) / (8 * 1024)

def filter1_crossbar_ok(cell: Dict[str, str], cap_kb: int, ww_bits: int) -> bool:
    access = cell.get("AccessType", "none").strip().lower()
    if access != "none": return True
    return cap_kb < crossbar_capacity_ceiling_kb(cell, ww_bits)

def filter2_high_ron_ok(cell: Dict[str, str], cap_kb: int) -> bool:
    ron = float(cell.get("ResistanceOnAtSetVoltage (ohm)", 0))
    return not (ron > 1e6 and cap_kb > 8 * 1024)



# ── Worker ────────────────────────────────────────────────────────────────────

def run_single_simulation(args: tuple) -> bool:
    """Worker executed in a parallel process pool."""
    (cap_kb, cell_path, variant_name, roadmap, base_cfg_content,
     temp_dir, results_dir, cfg_overrides, cfg_suffix,
     mem_type) = args

    final_csv = os.path.join(
        results_dir,
        f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}.csv"
    )
    if os.path.exists(final_csv):
        return True

    # Build cfg: Full exploration + Pruning → 8 Pareto rows per CSV
    new_cfg = re.sub(r"-Capacity\s*\(MB\):.*", f"-Capacity (KB): {cap_kb}", base_cfg_content)
    new_cfg = re.sub(r"-Capacity\s*\(KB\):.*", f"-Capacity (KB): {cap_kb}", new_cfg)
    abs_cell = os.path.abspath(cell_path)
    new_cfg  = re.sub(r"-MemoryCellInputFile:.*", f"-MemoryCellInputFile: {abs_cell}", new_cfg)

    # Force full exploration with pruning
    new_cfg  = re.sub(r"^[/-]*OptimizationTarget:.*", "", new_cfg, flags=re.MULTILINE)
    new_cfg  = re.sub(r"^[/-]*EnablePruning:.*",      "", new_cfg, flags=re.MULTILINE)
    new_cfg += "\n-OptimizationTarget: Full\n"
    new_cfg += "-EnablePruning: Yes\n"

    new_cfg = re.sub(
        r"^[/-]*DeviceRoadmap:.*", f"-DeviceRoadmap: {roadmap}",
        new_cfg, flags=re.MULTILINE
    )
    for param, value in cfg_overrides.items():
        pattern     = rf"^[/-]*{re.escape(param)}:.*"
        replacement = f"-{param}: {value}"
        if re.search(pattern, new_cfg, flags=re.MULTILINE):
            new_cfg = re.sub(pattern, replacement, new_cfg, flags=re.MULTILINE)
        else:
            new_cfg += f"\n-{param}: {value}\n"

    cfg_filename = f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}.cfg"
    cfg_filepath = os.path.join(temp_dir, cfg_filename)
    with open(cfg_filepath, 'w') as f:
        f.write(new_cfg)

    try:
        res = subprocess.run(
            ["./destiny", cfg_filepath],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res.returncode != 0:
            return False

        expected_csv = cfg_filepath.replace(".cfg", ".csv")
        if not os.path.exists(expected_csv):
            return False

        # CSV has proper headers from Result::printCsvHeader.
        # Add variant_name, then filter to the agreed column set.
        df = pd.read_csv(expected_csv)
        if df.empty:
            return False
        df.insert(0, "variant_name", variant_name)
        
        # Load cell parameters and append
        cell_params = parse_cell_params(cell_path)
        for k, v in cell_params.items():
            try:
                df[f"CellInput_{k}"] = float(v)
            except ValueError:
                df[f"CellInput_{k}"] = v
                
        # Keep only swept inputs + cache-level PPA + structural features + CellInput_*
        base_cols = [c for c in KEEP_COLS if c in df.columns]
        cell_cols = [c for c in df.columns if c.startswith("CellInput_")]
        df[base_cols + cell_cols].to_csv(final_csv, index=False)
        os.remove(expected_csv)
        return True

    finally:
        if os.path.exists(cfg_filepath):
            os.remove(cfg_filepath)

# ── Simulation Argument Builder ───────────────────────────────────────────────

def build_simulation_args(
    mem_type: str,
    cells: List[str],
    cell_dir: str,
    base_cfg_content: str,
    temp_dir: str,
    results_dir: str,
) -> Tuple[List[tuple], Dict[str, int]]:
    """
    Applies physics-aware Level 3/4/5 filters to produce the full argument list.
    """
    simulation_args: List[tuple] = []
    rejected: Dict[str, int] = {"L5_F1_crossbar": 0, "L5_F2_high_ron": 0}
    roadmaps = ROADMAPS.get(mem_type, ["HP"])

    for cell_file in cells:
        cell_path    = os.path.join(cell_dir, cell_file)
        variant_name = cell_file.split('.')[0]
        process_node = int(extract_process_node(cell_file))
        cell_params  = parse_cell_params(cell_path)

        for cap_kb in CAPACITY_SWEEP_KB:
            valid_stack_counts = [1]

            for stacked in valid_stack_counts:
                temperature = TEMPERATURE_FROM_STACK[mem_type][stacked]
                cfg_base = {
                    "Temperature (K)": temperature,
                    "StackedDieCount":  stacked,
                    "ProcessNode":      process_node,
                }

                # ── Cache mode: SRAM / eDRAM ──────────────────────────────────
                if mem_type in ("SRAM", "eDRAM"):
                    assoc = random.choice(capacity_bucket_associativities(cap_kb))
                    for ww_bits in capacity_bucket_word_widths(cap_kb):
                        overrides = dict(cfg_base)
                        overrides["WordWidth (bit)"]                = ww_bits
                        overrides["Associativity (for cache only)"] = assoc
                        if mem_type == "eDRAM":
                            rt_val = cell_params.get("RetentionTime (us)", 20.0)
                            try:
                                overrides["RetentionTime (us)"] = int(float(rt_val))
                            except ValueError:
                                overrides["RetentionTime (us)"] = rt_val

                        suffix = f"_ww{ww_bits}_a{assoc}_s{stacked}"
                        for roadmap in roadmaps:
                            simulation_args.append((
                                cap_kb, cell_path, variant_name, roadmap,
                                base_cfg_content, temp_dir, results_dir,
                                overrides, suffix, mem_type,
                            ))

                # ── RAM mode: RRAM ────────────────────────────────────────────
                elif mem_type == "RRAM":
                    ww_options      = capacity_bucket_word_widths(cap_kb)
                    sensing_options = rram_internal_sensing_options(cell_params)

                    for ww_bits in ww_options:
                        sensing_opt_cnt = len(sensing_options)
                        if not filter1_crossbar_ok(cell_params, cap_kb, ww_bits):
                            rejected["L5_F1_crossbar"] += sensing_opt_cnt
                            continue
                        if not filter2_high_ron_ok(cell_params, cap_kb):
                            rejected["L5_F2_high_ron"] += sensing_opt_cnt
                            continue

                        for sensing in sensing_options:
                            overrides = dict(cfg_base)
                            overrides["WordWidth (bit)"] = ww_bits
                            overrides["InternalSensing"] = "true" if sensing else "false"
                            suffix = (
                                f"_ww{ww_bits}"
                                f"_sens{'T' if sensing else 'F'}"
                                f"_s{stacked}"
                            )
                            for roadmap in roadmaps:
                                simulation_args.append((
                                    cap_kb, cell_path, variant_name, roadmap,
                                    base_cfg_content, temp_dir, results_dir,
                                    overrides, suffix, mem_type,
                                ))

    return simulation_args, rejected

# ── Main Orchestrator ─────────────────────────────────────────────────────────

def collect_simulations(mem_type: str) -> Tuple[List[tuple], Dict[str, int]]:
    """Gathers all simulation arguments and returns rejection stats."""
    temp_dir, results_dir = setup_dirs(mem_type, is_arch=False)

    base_cfg_file = CFG_TEMPLATES.get(mem_type)
    if not base_cfg_file:
        print(f"Error: Unsupported type '{mem_type}'")
        return [], {}

    with open(base_cfg_file, 'r') as f:
        base_cfg_content = f.read()

    cell_dir = f"synthetic_cells/{mem_type}"
    if not os.path.exists(cell_dir):
        print(f"No synthetic cells found for {mem_type}. Run generate_cells.py first.")
        return [], {}

    cells = sorted(f for f in os.listdir(cell_dir) if f.endswith('.cell'))
    if len(CAPACITY_SWEEP_KB) == 1:
        cells = cells[:1]
        
    simulation_args, rejected = build_simulation_args(
        mem_type, cells, cell_dir, base_cfg_content, temp_dir, results_dir
    )
    print(f"  Valid: {len(simulation_args)} | Rejected: {sum(rejected.values())}")
    return simulation_args, rejected


def execute_simulations(simulation_args: List[tuple], label: str):
    """Executes a list of simulation arguments in parallel."""
    if not simulation_args:
        return
    print(f"Launching {len(simulation_args)} parallel simulations for {label}...")
    run_count = success_count = 0
    total_runs = len(simulation_args)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_simulation, arg): arg for arg in simulation_args}
        for future in as_completed(futures):
            run_count += 1
            if future.result():
                success_count += 1
            if run_count % 1000 == 0 or run_count == total_runs:
                print(f"PROGRESS: {run_count}/{total_runs} completed")

    print(f"Success: {success_count}/{total_runs} simulations saved.")


def generate_and_run(mem_type: str):
    if not os.path.exists("./destiny"):
        print("Error: 'destiny' binary not found. Run 'make' first.")
        return
    sim_args, _ = collect_simulations(mem_type)
    execute_simulations(sim_args, mem_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physics-constrained DESTINY design-space exploration (Full Optimization)."
    )
    parser.add_argument(
        "--type", type=str, default="ALL",
        help="Memory type to explore (SRAM, RRAM, eDRAM, ALL)."
    )
    parser.add_argument(
        "--mini", action="store_true",
        help="Run a mini batch for testing pipeline (restrict sweep space)."
    )
    args = parser.parse_args()

    if args.mini:
        CAPACITY_SWEEP_KB = [64]

    if args.type.upper() == "ALL":
        all_sim_args = []
        for mem in CFG_TEMPLATES.keys():
            sim_args, _ = collect_simulations(mem)
            all_sim_args.extend(sim_args)
        if all_sim_args:
            execute_simulations(all_sim_args, "ALL")
        else:
            print("No simulations to run for ALL mode.")
    else:
        generate_and_run(args.type)

    print("\nExploration Sweep Done.")

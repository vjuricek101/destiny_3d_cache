#!/usr/bin/env python3
import os
import re
import subprocess
import argparse
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# ── Optimization Targets ──────────────────────────────────────────────────────
# 4 Pareto-diverse single-target runs per configuration.
OPTIMIZATION_TARGETS: List[str] = ["ReadLatency", "WriteEDP", "Area", "LeakagePower"]

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

# ── Cell File Utilities ───────────────────────────────────────────────────────

def parse_cell_params(filepath: str) -> Dict[str, str]:
    """Lightweight parser for DESTINY .cell files."""
    params: Dict[str, str] = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'): continue
            if line.startswith('-'):
                parts = line[1:].split(':', 1)
                if len(parts) == 2:
                    params[parts[0].strip()] = parts[1].strip()
    return params

def extract_process_node(cell_filename: str) -> Optional[int]:
    m = re.search(r'_n(\d+)\.cell$', cell_filename)
    return int(m.group(1)) if m else None

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

# ── Directory Setup ───────────────────────────────────────────────────────────

def setup_dirs(mem_type: str) -> Tuple[str, str]:
    temp_dir    = f"/dev/shm/vjuricek_destiny_tmp/{mem_type}"
    results_dir = f"exploration_results/{mem_type}"
    for d in [temp_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    return temp_dir, results_dir

# ── Unit Conversion Helpers ───────────────────────────────────────────────────

def _to_ns(val: float, unit: str) -> float:
    return {"ps": val / 1e3, "ns": val, "us": val * 1e3, "ms": val * 1e6}.get(unit, val)

def _to_nJ(val: float, unit: str) -> float:
    return {"pJ": val / 1e3, "nJ": val, "uJ": val * 1e3, "mJ": val * 1e6}.get(unit, val)

def _to_mW(val: float, unit: str) -> float:
    return {"uW": val / 1e3, "mW": val, "W": val * 1e3}.get(unit, val)

def _to_mm2(val: float, unit: str) -> float:
    return {"nm^2": val * 1e-12, "um^2": val * 1e-6, "mm^2": val, "m^2": val * 1e6}.get(unit, val)

# ── DESTINY Stdout Parser ─────────────────────────────────────────────────────

def parse_destiny_stdout(stdout: str) -> dict:
    """
    Parse a single-target DESTINY stdout.

    Returns a flat dict with:
      - Cache-level PPA  (cache mode: SRAM/eDRAM)  OR  array-level PPA (RAM mode: RRAM)
      - Internal floorplan params (bank/mat/subarray organization, mux levels, bandwidth)

    All latencies in ns, energies in nJ (cache) or pJ (array), power in mW, area in mm^2.
    Returns {} if DESTINY found no valid solutions (stdout has no configuration details).
    """
    result: dict = {}
    is_cache = "CACHE DESIGN -- SUMMARY" in stdout

    # Isolate DATA ARRAY section (strip tag array if present)
    data_section = stdout
    tag_marker = "CACHE TAG ARRAY DETAILS"
    if tag_marker in stdout:
        data_section = stdout[:stdout.index(tag_marker)]

    # ── Cache-level PPA ───────────────────────────────────────────────────────
    if is_cache:
        m = re.search(r"- Total Area = ([\d.]+)(mm\^2|um\^2|nm\^2)", stdout)
        if m:
            result["cache_area_mm2"] = _to_mm2(float(m.group(1)), m.group(2))

        m = re.search(r"Cache Hit Latency\s*=\s*([\d.]+)(ns|ps|us|ms)", stdout)
        if m:
            result["cache_hit_latency_ns"] = _to_ns(float(m.group(1)), m.group(2))

        m = re.search(r"Cache Miss Latency\s*=\s*([\d.]+)(ns|ps|us|ms)", stdout)
        if m:
            result["cache_miss_latency_ns"] = _to_ns(float(m.group(1)), m.group(2))

        m = re.search(r"Cache Write Latency\s*=\s*([\d.]+)(ns|ps|us|ms)", stdout)
        if m:
            result["cache_write_latency_ns"] = _to_ns(float(m.group(1)), m.group(2))

        m = re.search(r"Cache Hit Dynamic Energy\s*=\s*([\d.]+)(nJ|pJ|uJ|mJ)", stdout)
        if m:
            result["cache_hit_energy_nJ"] = _to_nJ(float(m.group(1)), m.group(2))

        m = re.search(r"Cache Miss Dynamic Energy\s*=\s*([\d.]+)(nJ|pJ|uJ|mJ)", stdout)
        if m:
            result["cache_miss_energy_nJ"] = _to_nJ(float(m.group(1)), m.group(2))

        m = re.search(r"Cache Write Dynamic Energy\s*=\s*([\d.]+)(nJ|pJ|uJ|mJ)", stdout)
        if m:
            result["cache_write_energy_nJ"] = _to_nJ(float(m.group(1)), m.group(2))

        m = re.search(r"Cache Total Leakage Power\s*=\s*([\d.]+)(mW|W|uW)", stdout)
        if m:
            result["cache_leakage_mW"] = _to_mW(float(m.group(1)), m.group(2))

    # ── Array-level PPA (RAM mode, or supplemental for cache mode) ────────────
    m = re.search(r"-\s*Read Latency\s*=\s*([\d.]+)(ns|ps|us|ms)", data_section)
    if m:
        result["read_latency_ns"] = _to_ns(float(m.group(1)), m.group(2))

    m = re.search(r"- Write Latency\s*=\s*([\d.]+)(ns|ps|us|ms)", data_section)
    if m:
        result["write_latency_ns"] = _to_ns(float(m.group(1)), m.group(2))

    m = re.search(r"-\s*Read Dynamic Energy\s*=\s*([\d.]+)(pJ|nJ|uJ|mJ)", data_section)
    if m:
        result["read_energy_pJ"] = _to_nJ(float(m.group(1)), m.group(2)) * 1e3

    m = re.search(r"- Write Dynamic Energy\s*=\s*([\d.]+)(pJ|nJ|uJ|mJ)", data_section)
    if m:
        result["write_energy_pJ"] = _to_nJ(float(m.group(1)), m.group(2)) * 1e3

    m = re.search(r"- Leakage Power\s*=\s*([\d.]+)(mW|W|uW)", data_section)
    if m:
        result["leakage_mW"] = _to_mW(float(m.group(1)), m.group(2))

    # ── Internal floorplan params (both modes) ────────────────────────────────
    m = re.search(r"Bank Organization:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", data_section)
    if m:
        result["destiny_bank_rows"]    = int(m.group(1))
        result["destiny_bank_cols"]    = int(m.group(2))
        result["destiny_bank_stacked"] = int(m.group(3))
        result["destiny_total_banks"]  = int(m.group(1)) * int(m.group(2)) * int(m.group(3))

    m = re.search(r"Mat Organization:\s*(\d+)\s*x\s*(\d+)", data_section)
    if m:
        result["destiny_mat_rows"]   = int(m.group(1))
        result["destiny_mat_cols"]   = int(m.group(2))
        result["destiny_total_mats"] = int(m.group(1)) * int(m.group(2))

    m = re.search(r"Subarray Size\s*:\s*(\d+)\s*Rows?\s*x\s*(\d+)\s*Columns?", data_section)
    if m:
        result["destiny_subarray_rows"] = int(m.group(1))
        result["destiny_subarray_cols"] = int(m.group(2))

    m = re.search(r"Row Activation\s*:\s*(\d+)\s*/\s*(\d+)", data_section)
    if m:
        result["destiny_row_activation_num"]   = int(m.group(1))
        result["destiny_row_activation_denom"] = int(m.group(2))

    m = re.search(r"Column Activation\s*:\s*(\d+)\s*/\s*(\d+)", data_section)
    if m:
        result["destiny_col_activation_num"]   = int(m.group(1))
        result["destiny_col_activation_denom"] = int(m.group(2))

    m = re.search(r"Senseamp Mux\s*:\s*(\d+)", data_section)
    if m:
        result["destiny_senseamp_mux"] = int(m.group(1))

    m = re.search(r"Output Level-2 Mux\s*:\s*(\d+)", data_section)
    if m:
        result["destiny_output_mux_l2"] = int(m.group(1))

    m = re.search(r"Read Bandwidth\s*=\s*([\d.]+)(GB/s|MB/s)", data_section)
    if m:
        bw = float(m.group(1))
        result["destiny_read_bw_GBs"] = bw if m.group(2) == "GB/s" else bw / 1000.0

    return result

# ── Worker ────────────────────────────────────────────────────────────────────

def run_single_simulation(args: tuple) -> bool:
    """Worker executed in a parallel process pool."""
    (cap_kb, cell_path, variant_name, roadmap, base_cfg_content,
     temp_dir, results_dir, cfg_overrides, cfg_suffix,
     mem_type, opt_target) = args

    final_csv = os.path.join(
        results_dir,
        f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}_opt_{opt_target}.csv"
    )
    if os.path.exists(final_csv):
        return True

    # Build cfg
    new_cfg = re.sub(r"-Capacity\s*\(MB\):.*", f"-Capacity (KB): {cap_kb}", base_cfg_content)
    new_cfg = re.sub(r"-Capacity\s*\(KB\):.*", f"-Capacity (KB): {cap_kb}", new_cfg)
    abs_cell = os.path.abspath(cell_path)
    new_cfg  = re.sub(r"-MemoryCellInputFile:.*", f"-MemoryCellInputFile: {abs_cell}", new_cfg)

    # Single-target optimization (NOT Full)
    new_cfg  = re.sub(r"^[/-]*OptimizationTarget:.*", "", new_cfg, flags=re.MULTILINE)
    new_cfg += f"\n-OptimizationTarget: {opt_target}\n"

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

    cfg_filename = (
        f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}_opt_{opt_target}.cfg"
    )
    cfg_filepath = os.path.join(temp_dir, cfg_filename)
    with open(cfg_filepath, 'w') as f:
        f.write(new_cfg)

    try:
        res = subprocess.run(
            ["./destiny", cfg_filepath],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if res.returncode != 0:
            return False

        parsed = parse_destiny_stdout(res.stdout)
        # Require at least one PPA field; reject "No valid solutions" runs
        ppa_keys = {"cache_hit_latency_ns", "cache_area_mm2",
                    "read_latency_ns", "leakage_mW"}
        if not ppa_keys.intersection(parsed.keys()):
            return False

        # Build self-describing row: swept inputs + PPA + floorplan
        row = {
            "mem_type":     mem_type,
            "variant_name": variant_name,
            "cap_kb":       cap_kb,
            "roadmap":      roadmap,
            "opt_target":   opt_target,
            **cfg_overrides,
            **parsed,
        }
        pd.DataFrame([row]).to_csv(final_csv, index=False)
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
    Each (config, opt_target) pair is one entry → 4 entries per base configuration.
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
                            for opt_target in OPTIMIZATION_TARGETS:
                                simulation_args.append((
                                    cap_kb, cell_path, variant_name, roadmap,
                                    base_cfg_content, temp_dir, results_dir,
                                    overrides, suffix, mem_type, opt_target,
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
                                for opt_target in OPTIMIZATION_TARGETS:
                                    simulation_args.append((
                                        cap_kb, cell_path, variant_name, roadmap,
                                        base_cfg_content, temp_dir, results_dir,
                                        overrides, suffix, mem_type, opt_target,
                                    ))

    return simulation_args, rejected

# ── Main Orchestrator ─────────────────────────────────────────────────────────

def collect_simulations(mem_type: str) -> Tuple[List[tuple], Dict[str, int]]:
    """Gathers all simulation arguments and returns rejection stats."""
    temp_dir, results_dir = setup_dirs(mem_type)

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
        description="Physics-constrained DESTINY design-space exploration (single-target stdout mode)."
    )
    parser.add_argument(
        "--type", type=str, default="ALL",
        help="Memory type to explore (SRAM, RRAM, eDRAM, ALL)."
    )
    args = parser.parse_args()

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

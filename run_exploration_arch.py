#!/usr/bin/env python3
import os
import re
import subprocess
import shutil
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

# ── Physics Tables ─────────────────────────────────────────────────────────────

# Global Temperature points for independent thermal sweep (K)
TEMPERATURE_SWEEP: List[int] = [300, 350, 400, 450]

BITLINE_LEAKAGE_TOLERANCE = 1
MAX_WORKERS: int = 64 # or os.cpu_count() 

# ── Architectural Design Space ────────────────────────────────────────────────

# Granular capacity sweep: 2KB to 32MB
CAPACITY_SWEEP_KB: List[int] = [2**i for i in range(1, 16)]

# Expanded architectural candidates (Power-of-2 only for DESTINY compatibility)
WORD_WIDTHS: List[int]     = [64, 128, 256, 512, 1024, 2048]
ASSOCIATIVITIES: List[int] = [1, 2, 4, 8, 16, 32, 64]
STACK_COUNTS: List[int]    = [1, 2, 4, 8, 16]

# ── Config Templates & Roadmaps ───────────────────────────────────────────────
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

# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_cell_params(filepath: str) -> Dict[str, str]:
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

def setup_dirs(mem_type: str) -> Tuple[str, str]:
    temp_dir    = f"/dev/shm/vjuricek_destiny_tmp/{mem_type}_arch"
    results_dir = f"exploration_results/{mem_type}_arch"
    for d in [temp_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    return temp_dir, results_dir


# ── DESTINY stdout parser ─────────────────────────────────────────────────────

def parse_destiny_stdout(stdout: str) -> dict:
    """
    Parse DESTINY stdout, extracting DATA ARRAY floorplan parameters as a flat dict. 
    Latencies in nanoseconds.
    """
    # Isolate data array section
    data_section = stdout
    tag_marker = "CACHE TAG ARRAY DETAILS"
    if tag_marker in stdout:
        data_section = stdout[:stdout.index(tag_marker)]

    result = {}

    def to_ns(val: str, unit: str) -> float:
        """Convert a latency value to nanoseconds."""
        v = float(val)
        return v / 1000.0 if unit == "ps" else v

    # ── Bank organization: "Bank Organization: 1 x 1 x 2"
    m = re.search(r"Bank Organization:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", data_section)
    if m:
        result["destiny_bank_rows"]    = int(m.group(1))
        result["destiny_bank_cols"]    = int(m.group(2))
        result["destiny_bank_stacked"] = int(m.group(3))
        result["destiny_total_banks"]  = (
            int(m.group(1)) * int(m.group(2)) * int(m.group(3))
        )

    # ── Mat organization: "Mat Organization: 2 x 2"
    m = re.search(r"Mat Organization:\s*(\d+)\s*x\s*(\d+)", data_section)
    if m:
        result["destiny_mat_rows"]   = int(m.group(1))
        result["destiny_mat_cols"]   = int(m.group(2))
        result["destiny_total_mats"] = int(m.group(1)) * int(m.group(2))

    # ── Subarray size: "Subarray Size    : 1024 Rows x 2048 Columns"
    m = re.search(
        r"Subarray Size\s*:\s*(\d+)\s*Rows?\s*x\s*(\d+)\s*Columns?",
        data_section
    )
    if m:
        result["destiny_subarray_rows"] = int(m.group(1))
        result["destiny_subarray_cols"] = int(m.group(2))

    # ── Row activation fraction: "Row Activation   : 1 / 2"
    m = re.search(r"Row Activation\s*:\s*(\d+)\s*/\s*(\d+)", data_section)
    if m:
        result["destiny_row_activation_num"]   = int(m.group(1))
        result["destiny_row_activation_denom"] = int(m.group(2))

    # ── Column activation fraction: "Column Activation: 1 / 1 x 1"
    m = re.search(r"Column Activation\s*:\s*(\d+)\s*/\s*(\d+)", data_section)
    if m:
        result["destiny_col_activation_num"]   = int(m.group(1))
        result["destiny_col_activation_denom"] = int(m.group(2))

    # ── Senseamp Mux level: "Senseamp Mux      : 1"
    m = re.search(r"Senseamp Mux\s*:\s*(\d+)", data_section)
    if m:
        result["destiny_senseamp_mux"] = int(m.group(1))

    # ── Output Level-2 Mux (main output mux depth)
    m = re.search(r"Output Level-2 Mux\s*:\s*(\d+)", data_section)
    if m:
        result["destiny_output_mux_l2"] = int(m.group(1))

    # ── Bandwidth
    m = re.search(r"Read Bandwidth\s*=\s*([\d.]+)(GB/s|MB/s)", data_section)
    if m:
        bw = float(m.group(1))
        result["destiny_read_bw_GBs"] = bw if m.group(2) == "GB/s" else bw / 1000.0

    return result


# ── Worker Function ────────────────────────────────────────────────────────────

def run_single_simulation(args: tuple) -> bool:
    (cap_kb, cell_path, variant_name, roadmap, base_cfg_content,
     temp_dir, results_dir, cfg_overrides, cfg_suffix) = args
    
    # adding skip logic so doesn't restart sweep from beginning
    final_csv = os.path.join(results_dir, f"{variant_name}_cap_{cap_kb}_rm_{roadmap}{cfg_suffix}.csv")
    if os.path.exists(final_csv):
        return True

    new_cfg = re.sub(r"-Capacity\s*\(MB\):.*", f"-Capacity (KB): {cap_kb}", base_cfg_content)
    new_cfg = re.sub(r"-Capacity\s*\(KB\):.*", f"-Capacity (KB): {cap_kb}", new_cfg)

    abs_cell = os.path.abspath(cell_path)
    new_cfg  = re.sub(r"-MemoryCellInputFile:.*", f"-MemoryCellInputFile: {abs_cell}", new_cfg)
    new_cfg  = re.sub(r"^[/-]*OptimizationTarget:.*", "", new_cfg, flags=re.MULTILINE)
    new_cfg += "\n-OptimizationTarget: Full\n"
    new_cfg  = re.sub(
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
            stdout=subprocess.PIPE,      # capture stdout for floorplan parsing
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if res.returncode != 0:
            return False

        expected_csv = cfg_filepath.replace(".cfg", ".csv")
        if not os.path.exists(expected_csv):
            return False

        # Parse internal floorplan parameters from DESTINY's text report
        # and append them as additional columns to the output CSV.
        floorplan = parse_destiny_stdout(res.stdout)
        if floorplan:
            # Read DESTINY's raw output (no header)
            df_out = pd.read_csv(expected_csv, header=None)
            for k, v in floorplan.items():
                df_out[k] = v
            # Save with header
            df_out.to_csv(expected_csv, index=False)

        final_csv = os.path.join(
            results_dir, cfg_filename.replace(".cfg", ".csv")
        )
        shutil.move(expected_csv, final_csv)
        return True

    finally:
        if os.path.exists(cfg_filepath):
            os.remove(cfg_filepath)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def collect_simulations(mem_type: str) -> List[tuple]:
    temp_dir, results_dir = setup_dirs(mem_type)
    base_cfg_file = CFG_TEMPLATES.get(mem_type)
    with open(base_cfg_file, 'r') as f:
        base_cfg_content = f.read()

    cell_dir = f"synthetic_cells/{mem_type}_arch"
    if not os.path.exists(cell_dir):
        print(f"No architectural cells found for {mem_type}. Run generate_cells_arch.py first.")
        return []

    cells    = sorted(f for f in os.listdir(cell_dir) if f.endswith('.cell'))
    roadmaps = ROADMAPS.get(mem_type, ["HP"])
    simulation_args = []

    for cell_file in cells:
        cell_path    = os.path.join(cell_dir, cell_file)
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
                                    overrides["RetentionTime (us)"] = int(
                                        float(cell_params.get("RetentionTime (us)", 20))
                                    )
                                suffix = f"_ww{ww_bits}_a{assoc}_s{stacked}_t{temperature}"
                                for roadmap in roadmaps:
                                    simulation_args.append((
                                        cap_kb, cell_path, variant_name, roadmap,
                                        base_cfg_content, temp_dir, results_dir,
                                        overrides, suffix,
                                    ))

                    elif mem_type == "RRAM":
                        for ww_bits in WORD_WIDTHS:
                            for sensing in [True, False]:
                                overrides = dict(cfg_base)
                                overrides["WordWidth (bit)"] = ww_bits
                                overrides["InternalSensing"] = "true" if sensing else "false"
                                suffix = f"_ww{ww_bits}_sens{'T' if sensing else 'F'}_s{stacked}_t{temperature}"
                                for roadmap in roadmaps:
                                    simulation_args.append((
                                        cap_kb, cell_path, variant_name, roadmap,
                                        base_cfg_content, temp_dir, results_dir,
                                        overrides, suffix,
                                    ))

    return simulation_args


def execute_simulations(simulation_args: List[tuple], label: str):
    if not simulation_args:
        return
    print(f"\nLaunching {len(simulation_args)} architectural sweep simulations for {label}...")
    run_count = success_count = 0
    total_runs = len(simulation_args)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single_simulation, arg): arg
            for arg in simulation_args
        }
        for future in as_completed(futures):
            run_count += 1
            if future.result():
                success_count += 1
            if run_count % 500 == 0 or run_count == total_runs:
                print(f"  [{run_count}/{total_runs}] completed ({success_count} outputs).")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dense Cartesian architectural sweep for DESTINY."
    )
    parser.add_argument("--type",     type=str, default="ALL",
                        help="Memory technology to sweep: SRAM, RRAM, eDRAM, or ALL")
    args = parser.parse_args()

    types = (
        ["SRAM", "RRAM", "eDRAM"]
        if args.type.upper() == "ALL"
        else [args.type.upper()]
    )
    for t in types:
        sim_args = collect_simulations(t)
        execute_simulations(sim_args, t + "_arch")

    print("\nArchitectural Sweep Done.")


if __name__ == "__main__":
    main()
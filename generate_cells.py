#!/usr/bin/env python3
import argparse
import math
import os
import random
import shutil
import sys
from typing import Dict, Any, List, Optional

# Physics Tables
VDD_TABLE: Dict[int, float] = {22: 0.9, 32: 1.0, 45: 1.1, 65: 1.2} # for RRAM high-boost-ratio check.
VDD_EDRAM: Dict[int, float] = {65: 1.2, 45: 1.1, 32: 1.0}
CCELL_MAX: Dict[int, float] = {65: 30e-15, 45: 25e-15, 32: 20e-15}
RETENTION_MAX: Dict[int, float] = {65: 40.0, 45: 30.0, 32: 20.0}
I_OFF_PER_METER: Dict[int, float] = {22: 100e-9, 32: 50e-9, 45: 20e-9, 65: 5e-9} # Used for RetentionTime vs Ccell/leakage consistency.

# Configuration
MEMORY_CONFIGS = {
    "SRAM": {
        "BASE_CELL_FILE": "config/sample_SRAM.cell",  # 65nm baseline
        "VALID_NODES": [22, 32, 45, 65],
        "VARIATION_RANGES": {
            # CellArea is derived from widths — not sampled directly.
            # MinSenseVoltage is derived from AccessCMOSWidth — not sampled directly.
            "SRAMCellNMOSWidth (F)": (2.0, 3.5),
            "SRAMCellPMOSWidth (F)": (1.5, 3.0),
            "AccessCMOSWidth (F)":   (2.0, 3.5),
            "CellAspectRatio":       (0.8, 2.0),
        }
    },
    "RRAM": {
        "BASE_CELL_FILE": "config/sample_RRAM.cell",
        "VALID_NODES": [22, 32, 45, 65],
        "VARIATION_RANGES": {
            # CellArea is derived from AccessType (CMOS vs diode/none).
            # SetPulse is derived from Ron and SetVoltage — not sampled directly.
            "ResistanceOnAtSetVoltage (ohm)":  (5_000,   50_000),
            "ResistanceOffAtSetVoltage (ohm)": (100_000, 5_000_000),
            "ReadVoltage (V)":                 (0.2, 0.8),
            "ResetVoltage (V)":                (2.0, 5.0),
            "SetVoltage (V)":                  (2.0, 5.0),
            "ResetPulse (ns)":                 (10.0, 50.0),
            "VoltageDropAccessDevice (V)":     (0.05, 0.4),
        }
    },
    "eDRAM": {
        "BASE_CELL_FILE": "config/sample_2D_eDRAM.cell",
        # physically invalid at 22nm 
        "VALID_NODES": [32, 45, 65],
        "VARIATION_RANGES": {
            # CellArea is derived from AccessWidth and Capacitance.
            # MinSenseVoltage is derived from DRAMCellCapacitance — not sampled directly.
            "CellAspectRatio":         (0.8, 2.5),
            "AccessCMOSWidth (F)":     (1.0, 2.5),
            # Broad range; per-node CCELL_MAX is enforced at sample time.
            "DRAMCellCapacitance (F)": (5e-15, 30e-15),
            # Broad range; RETENTION_MAX and leakage consistency enforced in valid_edram.
            "RetentionTime (us)":      (5, 50),
        }
    }
}


# Utility Functions

def log_uniform(min_val: float, max_val: float) -> float:
    """Samples a value from a log-uniform distribution."""
    return 10 ** random.uniform(math.log10(min_val), math.log10(max_val))


def parse_cell_file(filepath: str) -> Dict[str, str]:
    """Parses a DESTINY .cell file into a dictionary of parameters."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('-'):
                key_part, val_part = line[1:].split(':', 1)
                params[key_part.strip()] = val_part.strip()
    return params


def write_cell_file(params: Dict[str, Any], filepath: str):
    """Writes parameters to a DESTINY .cell file."""
    with open(filepath, 'w') as f:
        f.write("// SYNTHETIC CELL GENERATED FOR DESIGN SPACE EXPLORATION\n\n")

        # MemCellType must appear first for the DESTINY parser.
        type_key = "MemCellType"
        if type_key in params:
            f.write(f"-{type_key}: {params[type_key]}\n")

        # Write comment flags (prefixed with //) before the main parameters.
        comment_keys = [k for k in params if k.startswith("//")]
        for key in comment_keys:
            f.write(f"{key}: {params[key]}\n")

        for key, value in params.items():
            if key == type_key or key.startswith("//"):
                continue
            if isinstance(value, float):
                if value < 1e-6:
                    f.write(f"-{key}: {value:.2e}\n")
                else:
                    f.write(f"-{key}: {value:.4f}\n")
            else:
                f.write(f"-{key}: {value}\n")


# Per-Node Physics Helpers

def max_retention_from_cell(c_cell: float, access_width_f: float,
                             process_node: int) -> float:
    """
    Filter 4: maximum physically sustainable RetentionTime (us) given the
    cell's capacitance, access-transistor width, and node leakage.

    Physics:  t_ret = (C * Vdd/2) / I_leak
              I_leak = I_off_per_meter * W_access_m
    """
    i_off = I_OFF_PER_METER[process_node]
    w_access_m = access_width_f * process_node * 1e-9   # F × nm → m
    i_leak = i_off * w_access_m
    if i_leak <= 0:
        return float('inf')
    vdd = VDD_EDRAM[process_node]
    return (c_cell * vdd / 2.0) / i_leak * 1e6          # convert s → us


# Physics Constraint Validation

def valid_sram(p: Dict[str, Any]) -> bool:
    """Validates SRAM transistor sizing and sense-voltage compatibility."""
    try:
        wn    = float(p["SRAMCellNMOSWidth (F)"])
        wp    = float(p["SRAMCellPMOSWidth (F)"])
        wac   = float(p["AccessCMOSWidth (F)"])
        vsense = float(p["MinSenseVoltage (mV)"])

        # Stability: NMOS must be stronger than the access transistor.
        if wn / wac < 1.2:        return False
        # Writeability: PMOS mustn't overpower access.
        if wp / wac > 1.0:        return False
        # Consistency: stronger access enables lower sense-voltage thresholds.
        if vsense > 120 / wac:    return False

        return True
    except (KeyError, ZeroDivisionError, ValueError):
        return False


def valid_rram(p: Dict[str, Any]) -> bool:
    """Validates RRAM resistance ratios and programming pulses."""
    try:
        ron   = float(p["ResistanceOnAtSetVoltage (ohm)"])
        roff  = float(p["ResistanceOffAtSetVoltage (ohm)"])
        vset  = float(p["SetVoltage (V)"])
        pulse = float(p["SetPulse (ns)"])

        # Realistic resistance window (10× to 10 000×).
        if roff / ron < 10 or roff / ron > 1e4:  return False
        if pulse < 2 or pulse > 50:               return False
        if vset < 1.5 or vset > 5.0:             return False

        return True
    except (KeyError, ZeroDivisionError, ValueError):
        return False


def valid_edram(p: Dict[str, Any], process_node: int) -> bool:
    """
    Validates eDRAM cells against:
      - RetentionTime node ceiling (RETENTION_MAX).
      - RetentionTime vs Ccell/leakage consistency.
    """
    try:
        c    = float(p["DRAMCellCapacitance (F)"])
        wac  = float(p["AccessCMOSWidth (F)"])
        tret = float(p["RetentionTime (us)"])

        # node-level ceiling on retention time.
        node_ceiling = RETENTION_MAX.get(process_node, 40.0)
        if tret > node_ceiling:
            return False

        # leakage-limited retention ceiling (e.g. 5fF + 50us).
        t_max = max_retention_from_cell(c, wac, process_node)
        if tret > t_max:
            return False

        return True
    except (KeyError, ZeroDivisionError, ValueError):
        return False


# Generation Logic

def generate_synthetic_cells(mem_type: str, num_variants: int):
    """Generates synthetic variants for given technology."""
    if mem_type not in MEMORY_CONFIGS:
        print(f"Error: Unsupported memory type '{mem_type}'")
        return

    config     = MEMORY_CONFIGS[mem_type]
    base_file  = config["BASE_CELL_FILE"]
    ranges     = config["VARIATION_RANGES"]
    valid_nodes = config["VALID_NODES"]

    output_dir = f"synthetic_cells/{mem_type}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    base_params = parse_cell_file(base_file)
    if not base_params:
        sys.exit(f"Error: Could not parse baseline file {base_file}")

    print(f"Generating {num_variants} {mem_type} variants...")

    i, attempts   = 0, 0
    high_boost_count = 0
    max_attempts  = num_variants * 20

    while i < num_variants and attempts < max_attempts:
        attempts += 1
        new_params = base_params.copy()

        # Sample process node
        process_node = random.choice(valid_nodes)
        new_params["ProcessNode"] = process_node

        # Sample cell parameters
        for key, (min_val, max_val) in ranges.items():
            if any(term in key for term in ["Resistance", "Capacitance", "Energy"]):
                new_params[key] = log_uniform(min_val, max_val)
            else:
                new_params[key] = random.uniform(min_val, max_val)

        # Per-node clamping & FinFET quantization
        if mem_type == "SRAM":
            wn  = new_params["SRAMCellNMOSWidth (F)"]
            wp  = new_params["SRAMCellPMOSWidth (F)"]
            wac = new_params["AccessCMOSWidth (F)"]

            new_params["CellArea (F^2)"]        = 60 + 20 * (wn + wac) + 10 * wp
            # MinSenseVoltage derived from access width (not sampled independently).
            new_params["MinSenseVoltage (mV)"]  = (80 / wac) * random.uniform(0.8, 1.2)

        elif mem_type == "eDRAM":
            wac = new_params["AccessCMOSWidth (F)"]
            c   = new_params["DRAMCellCapacitance (F)"]

            # -- Fabrication ceiling: cap Ccell to what the node can physically build.
            node_ccell_max = CCELL_MAX[process_node]
            if c > node_ccell_max:
                c = random.uniform(5e-15, node_ccell_max)
                new_params["DRAMCellCapacitance (F)"] = c

            # -- RetentionTime ceiling: cap to the node's leakage-limited maximum.
            node_ret_max = RETENTION_MAX[process_node]
            tret = new_params["RetentionTime (us)"]
            if tret > node_ret_max:
                tret = random.uniform(5, node_ret_max)
                new_params["RetentionTime (us)"] = tret

            new_params["CellArea (F^2)"]       = 10 + 6 * wac + 1.2 * (c / 1e-15)
            # MinSenseVoltage: ~80mV for 20fF, scaled inversely with capacitance.
            new_params["MinSenseVoltage (mV)"] = (
                80 * (20e-15 / c) * random.uniform(0.8, 1.2)
            )

        elif mem_type == "RRAM":
            # Model different access device configurations.
            selected_access = random.choice(["CMOS", "diode", "none"])
            new_params["AccessType"] = selected_access

            if selected_access == "CMOS":
                new_params["CellArea (F^2)"] = random.uniform(30, 60)
            else:
                new_params["CellArea (F^2)"] = random.uniform(4, 12)

            # Link resistance parameters across different operating voltages.
            r_on = new_params.get("ResistanceOnAtSetVoltage (ohm)")
            if r_on:
                new_params["ResistanceOnAtResetVoltage (ohm)"]     = r_on
                new_params["ResistanceOnAtReadVoltage (ohm)"]      = r_on * random.uniform(1.1, 1.5)
                new_params["ResistanceOnAtHalfResetVoltage (ohm)"] = r_on * random.uniform(1.1, 1.3)

            r_off = new_params.get("ResistanceOffAtSetVoltage (ohm)")
            if r_off:
                new_params["ResistanceOffAtResetVoltage (ohm)"] = r_off
                new_params["ResistanceOffAtReadVoltage (ohm)"]  = r_off * random.uniform(0.8, 1.0)

            # SetPulse derived from Ron and SetVoltage — stronger drive + lower
            # resistance → faster switching.  Sampling independently would inject
            # random noise into the feature space.
            ron   = new_params["ResistanceOnAtSetVoltage (ohm)"]
            vset  = new_params["SetVoltage (V)"]
            pulse = (ron / (vset ** 2)) * random.uniform(0.5, 2.0)
            new_params["SetPulse (ns)"] = max(2, min(pulse, 50))

            # -- High-boost-ratio flag: Vset > 2×VDD requires a charge pump.
            # DESTINY models the overhead but does not reject the cell.
            # Flag these for downstream analysis (severe area/energy penalties).
            vdd = VDD_TABLE[process_node]
            if vset > 2.0 * vdd:
                new_params["//FLAG_HighBoostRatio"] = (
                    f"Vset={vset:.2f}V > 2xVDD={2*vdd:.2f}V at {process_node}nm"
                )
                high_boost_count += 1

        # Physics Validation
        valid = True
        if   mem_type == "SRAM":  valid = valid_sram(new_params)
        elif mem_type == "RRAM":  valid = valid_rram(new_params)
        elif mem_type == "eDRAM": valid = valid_edram(new_params, process_node)

        if not valid:
            continue

        # Write valid variant
        # Embed process node in filename for traceability in run_exploration.py.
        filename = f"synthetic_variant_{i}_n{process_node}.cell"
        filepath = os.path.join(output_dir, filename)
        write_cell_file(new_params, filepath)
        i += 1

    # Summary
    if i < num_variants:
        print(f"  Note: Only {i}/{num_variants} valid variants generated.")
    else:
        print(f"  Success: {i} variants written ({attempts} attempts).")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic memory cell configurations for DESTINY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--type", type=str, default="ALL",
        help="Memory type to generate (SRAM, RRAM, eDRAM, or ALL)."
    )
    parser.add_argument(
        "--num_variants", type=int, default=2000,
        help="Number of valid variants to generate per technology."
    )
    args = parser.parse_args()

    # No header print

    selected_types = (
        list(MEMORY_CONFIGS.keys()) if args.type.upper() == "ALL" else [args.type]
    )

    for mem in selected_types:
        generate_synthetic_cells(mem, args.num_variants)

    print("\nGeneration process complete.")


if __name__ == "__main__":
    main()
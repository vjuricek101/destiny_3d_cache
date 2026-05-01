#!/usr/bin/env python3
"""
generate_cells_arch.py — Denser Architectural Sweep

Keep physics parameters fixed at nominal values for each process node. 
"""

import argparse
import os
import shutil
from typing import Dict, Any

# Supply voltages per node (V)
VDD_TABLE: Dict[int, float] = {22: 0.9, 32: 1.0, 45: 1.1, 65: 1.2}

MEMORY_CONFIGS = {
    "SRAM": {
        "BASE_CELL_FILE": "config/sample_SRAM.cell",
        "VALID_NODES": [22, 32, 45, 65],
        "NOMINAL": {
            "SRAMCellNMOSWidth (F)": 2.5,
            "SRAMCellPMOSWidth (F)": 2.0,
            "AccessCMOSWidth (F)":   2.5,
            "CellAspectRatio":       1.5,
        }
    },
    "RRAM": {
        "BASE_CELL_FILE": "config/sample_RRAM.cell",
        "VALID_NODES": [22, 32, 45, 65],
        "NOMINAL": {
            "ResistanceOnAtSetVoltage (ohm)":  20_000,
            "ResistanceOffAtSetVoltage (ohm)": 500_000,
            "ReadVoltage (V)":                 0.4,
            "ResetVoltage (V)":                3.0,
            "SetVoltage (V)":                  3.0,
            "ResetPulse (ns)":                 20.0,
            "VoltageDropAccessDevice (V)":     0.15,
        }
    },
    "eDRAM": {
        "BASE_CELL_FILE": "config/sample_2D_eDRAM.cell",
        "VALID_NODES": [32, 45, 65],
        "NOMINAL": {
            "CellAspectRatio":         1.5,
            "AccessCMOSWidth (F)":     1.5,
            "DRAMCellCapacitance (F)": 15e-15,
            "RetentionTime (us)":      20,
        }
    }
}

def parse_cell_file(filepath: str) -> Dict[str, str]:
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
    with open(filepath, 'w') as f:
        f.write("// SYNTHETIC CELL GENERATED FOR ARCHITECTURAL SWEEP (FIXED PHYSICS)\n\n")
        type_key = "MemCellType"
        if type_key in params:
            f.write(f"-{type_key}: {params[type_key]}\n")

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

def generate_arch_cells(mem_type: str):
    if mem_type not in MEMORY_CONFIGS: return
    
    config = MEMORY_CONFIGS[mem_type]
    base_params = parse_cell_file(config["BASE_CELL_FILE"])
    output_dir = f"synthetic_cells/{mem_type}_arch"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"Generating architectural baseline cells for {mem_type} in {output_dir}...")

    for node in config["VALID_NODES"]:
        new_params = base_params.copy()
        new_params["ProcessNode"] = node
        
        # Apply nominal values
        for key, val in config["NOMINAL"].items():
            new_params[key] = val
        
        # Derived for SRAM
        if mem_type == "SRAM":
            wn, wp, wac = new_params["SRAMCellNMOSWidth (F)"], new_params["SRAMCellPMOSWidth (F)"], new_params["AccessCMOSWidth (F)"]
            new_params["CellArea (F^2)"] = 60 + 20 * (wn + wac) + 10 * wp
            new_params["MinSenseVoltage (mV)"] = 80 / wac
        
        elif mem_type == "eDRAM":
            wac = new_params["AccessCMOSWidth (F)"]
            c = new_params["DRAMCellCapacitance (F)"]
            new_params["CellArea (F^2)"] = 10 + 6 * wac + 1.2 * (c / 1e-15)
            new_params["MinSenseVoltage (mV)"] = 80 * (20e-15 / c)

        elif mem_type == "RRAM":
            new_params["AccessType"] = "CMOS"
            new_params["CellArea (F^2)"] = 45.0
            r_on = new_params["ResistanceOnAtSetVoltage (ohm)"]
            new_params["ResistanceOnAtResetVoltage (ohm)"]     = r_on
            new_params["ResistanceOnAtReadVoltage (ohm)"]      = r_on * 1.2
            new_params["ResistanceOnAtHalfResetVoltage (ohm)"] = r_on * 1.1
            r_off = new_params["ResistanceOffAtSetVoltage (ohm)"]
            new_params["ResistanceOffAtResetVoltage (ohm)"] = r_off
            new_params["ResistanceOffAtReadVoltage (ohm)"]  = r_off * 0.9
            new_params["SetPulse (ns)"] = 15.0

        filename = f"arch_variant_nominal_n{node}.cell"
        write_cell_file(new_params, os.path.join(output_dir, filename))
        print(f"  Written {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="ALL")
    args = parser.parse_args()
    
    types = ["SRAM", "RRAM", "eDRAM"] if args.type.upper() == "ALL" else [args.type]
    for t in types:
        generate_arch_cells(t)

if __name__ == "__main__":
    main()

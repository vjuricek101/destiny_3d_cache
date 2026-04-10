import os
import random
import shutil
import argparse
import math

def log_uniform(min_val, max_val):
    """Calculates a value based on a log-uniform distribution."""
    return 10 ** random.uniform(math.log10(min_val), math.log10(max_val))

# Memory Types and their configs
MEMORY_CONFIGS = {
    "SRAM": {
        "BASE_CELL_FILE": "config/sample_SRAM.cell",
        "VARIATION_RANGES": {
            "CellArea (F^2)": (120.0, 220.0),
            "SRAMCellNMOSWidth (F)": (1.0, 3.0),
            "SRAMCellPMOSWidth (F)": (1.0, 3.0),
            "AccessCMOSWidth (F)": (1.0, 3.0),
            "CellAspectRatio": (0.8, 2.0),
            "MinSenseVoltage (mV)": (60, 100)
        }
    },
    "RRAM": {
        "BASE_CELL_FILE": "config/sample_RRAM.cell",
        "VARIATION_RANGES": {
            "CellArea (F^2)": (4.0, 12.0), 
            "ResistanceOnAtSetVoltage (ohm)": (5000, 50000),
            "ResistanceOffAtSetVoltage (ohm)": (100000, 5000000),
            "ReadVoltage (V)": (0.2, 0.8),
            "ResetVoltage (V)": (2.0, 5.0),
            "SetVoltage (V)": (2.0, 5.0),
            "ResetPulse (ns)": (10.0, 60.0),
            "SetPulse (ns)": (10.0, 60.0),
            "VoltageDropAccessDevice (V)": (0.05, 0.5)
        }
    },
    "eDRAM": {
        "BASE_CELL_FILE": "config/eDRAM.cell",
        "VARIATION_RANGES": {
            "CellArea (F^2)": (30.0, 80.0),
            "CellAspectRatio": (0.8, 2.0),
            "AccessCMOSWidth (F)": (1.0, 2.5),
            "DRAMCellCapacitance (F)": (5e-15, 25e-15),
            "MinSenseVoltage (mV)": (5, 20)
        }
    }
}



def parse_cell_file(filepath):
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

def write_cell_file(params, filepath):
    """Writes parameters to a DESTINY .cell file."""
    with open(filepath, 'w') as f:
        f.write("// SYNTHETIC CELL GENERATED FOR DESIGN SPACE EXPLORATION\n\n")
        type_key = "MemCellType"
        if type_key in params:
            f.write(f"-{type_key}: {params[type_key]}\n")
            
        for key, value in params.items():
            if key != type_key:
                if isinstance(value, float):
                    if value < 1e-6:
                        f.write(f"-{key}: {value:.2e}\n")
                    else:
                        f.write(f"-{key}: {value:.4f}\n")
                else:
                    f.write(f"-{key}: {value}\n")

def generate_synthetic_cells(mem_type, num_variants):
    if mem_type not in MEMORY_CONFIGS:
        print(f"Error: Unsupported memory type '{mem_type}'")
        return

    config = MEMORY_CONFIGS[mem_type]
    base_file = config["BASE_CELL_FILE"]
    ranges = config["VARIATION_RANGES"]
    
    output_dir = f"synthetic_cells/{mem_type}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Loading baseline {mem_type} from: {base_file}")
    base_params = parse_cell_file(base_file)
    
    print(f"Generating {num_variants} synthetic variants for {mem_type} in {output_dir}...")
    for i in range(num_variants):
        new_params = base_params.copy()
        for key, (min_val, max_val) in ranges.items():
            if key in new_params:
                # Logarithmic sampling
                if "Resistance" in key or "Capacitance" in key:
                    new_params[key] = log_uniform(min_val, max_val)
                else:
                    new_params[key] = random.uniform(min_val, max_val)
        
        # RRAM logic for different Access Types
        if mem_type == "RRAM":
            # 1. Randomize AccessType (Logic: 1T1R = CMOS, 1S1R = diode, Crossbar = none)
            access_types = ["CMOS", "diode", "none"]
            selected_access = random.choice(access_types)
            new_params["AccessType"] = selected_access
            
            if selected_access == "CMOS":
                new_params["AccessCMOSWidth (F)"] = random.uniform(1.0, 4.0)
                if new_params["CellArea (F^2)"] < 30:
                    new_params["CellArea (F^2)"] = random.uniform(30, 53)
            else:
                if new_params["CellArea (F^2)"] > 12:
                    new_params["CellArea (F^2)"] = random.uniform(4, 12)

            # 2. Match related resistance parameters proportionally
            r_on = new_params.get("ResistanceOnAtSetVoltage (ohm)")
            if r_on:
                new_params["ResistanceOnAtResetVoltage (ohm)"] = r_on
                new_params["ResistanceOnAtReadVoltage (ohm)"] = r_on * random.uniform(1.1, 1.5)
                new_params["ResistanceOnAtHalfResetVoltage (ohm)"] = r_on * random.uniform(1.1, 1.3)
                
            r_off = new_params.get("ResistanceOffAtSetVoltage (ohm)")
            if r_off:
                new_params["ResistanceOffAtResetVoltage (ohm)"] = r_off
                new_params["ResistanceOffAtReadVoltage (ohm)"] = r_off * random.uniform(0.8, 1.0)

        filename = f"synthetic_variant_{i}.cell"
        filepath = os.path.join(output_dir, filename)
        write_cell_file(new_params, filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic memory cells.")
    parser.add_argument("--type", type=str, default="ALL", help="Memory type (SRAM, RRAM, eDRAM, or ALL)")
    parser.add_argument("--num_variants", type=int, default=200, help="Number of variants to generate per type")
    args = parser.parse_args()
    
    mem_types = MEMORY_CONFIGS.keys() if args.type == "ALL" else [args.type]
    
    for mem in mem_types:
        generate_synthetic_cells(mem, args.num_variants)
        
    print(f"\nStep 1 Complete: Synthetic Data Generation successful.")

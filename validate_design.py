#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np

def validate(tech, cap_kb, ww, assoc, stack, temp, wn, wp, wac, node=32, roadmap="HP"):
    """
    Validates an inverse design result by running it through the actual DESTINY engine.
    """
    print(f"DESTINY Validation for {tech}")
    print(f"Arch: {cap_kb}KB, WW={ww}, Assoc={assoc}, Stack={stack}, Temp={temp}K")
    print(f"Cell: wn={wn:.3f}, wp={wp:.3f}, wac={wac:.3f} | Node: {node}nm {roadmap}")

    # 1. Create .cell file
    cell_content = f"""
-MemCellType: {tech}
-ProcessNode: {node}
-SRAMCellNMOSWidth (F): {wn:.4f}
-SRAMCellPMOSWidth (F): {wp:.4f}
-AccessCMOSWidth (F): {wac:.4f}
-MinSenseVoltage (mV): {80.0/wac:.4f}
-CellArea (F^2): {60 + 20*(wn + wac) + 10*wp:.4f}
-CellAspectRatio: 1.5
"""
    cell_file = "validation_temp.cell"
    with open(cell_file, 'w') as f: f.write(cell_content)

    # 2. Create .cfg file
    cfg_content = f"""
-OptimizationTarget: Full
-Capacity (KB): {cap_kb}
-WordWidth (bit): {ww}
-Associativity (for cache only): {assoc}
-StackedDieCount: {stack}
-Temperature (K): {temp}
-DeviceRoadmap: {roadmap}
-MemoryCellInputFile: {os.path.abspath(cell_file)}
"""
    cfg_file = "validation_temp.cfg"
    with open(cfg_file, 'w') as f: f.write(cfg_content)

    # 3. Run DESTINY
    print("Running DESTINY...")
    try:
        res = subprocess.run(["./destiny", cfg_file], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"DESTINY Error: {res.stderr}")
            return
        
        csv_file = cfg_file.replace(".cfg", ".csv")
        if not os.path.exists(csv_file):
            print("Error: DESTINY CSV not generated.")
            return
        
        # 4. Parse Results
        df = pd.read_csv(csv_file)
        
        real_area = df["cache_area_mm2"].iloc[0]
        real_lat  = df["cache_hit_latency_ns"].iloc[0]
        real_en   = df["cache_hit_energy_nJ"].iloc[0]
        real_leak = df["cache_leakage_mW"].iloc[0]

        print("\n--- Physical Results (DESTINY) ---")
        print(f"  Latency: {real_lat:.4f} ns")
        print(f"  Area:    {real_area:.4f} mm^2")
        print(f"  Energy:  {real_en:.4f} nJ")
        print(f"  Leakage: {real_leak:.4f} mW")

    finally:
        # Cleanup
        for f in [cell_file, cfg_file, cfg_file.replace(".cfg", ".csv")]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tech",    default="SRAM")
    parser.add_argument("--cap",     type=int, required=True)
    parser.add_argument("--ww",      type=int, required=True)
    parser.add_argument("--assoc",   type=int, required=True)
    parser.add_argument("--stack",   type=int, required=True)
    parser.add_argument("--temp",    type=int, required=True)
    parser.add_argument("--wn",      type=float, required=True)
    parser.add_argument("--wp",      type=float, required=True)
    parser.add_argument("--wac",     type=float, required=True)
    parser.add_argument("--node",    type=int,   default=32)
    parser.add_argument("--roadmap", default="HP")
    args = parser.parse_args()

    validate(args.tech, args.cap, args.ww, args.assoc, args.stack, args.temp, 
             args.wn, args.wp, args.wac, node=args.node, roadmap=args.roadmap)

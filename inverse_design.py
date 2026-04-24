#!/usr/bin/env python3
import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from train_model import PPA_MLP

class UnifiedOptimizer:
    def __init__(self, tech, is_arch=False):
        self.tech = tech
        self.is_arch = is_arch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine paths
        suffix = "_arch" if is_arch else "_full"
        self.model_dir = f"model_output/{tech.lower()}{suffix}"
        
        # Load assets
        with open(os.path.join(self.model_dir, "feature_cols.json"), "r") as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(self.model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
            
        self.model = PPA_MLP(len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, "model.pt"), map_location=self.device))
        self.model.eval()

        self.means = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device)
        self.stds = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

    def _get_target_tensor(self, targets):
        t_vals = [targets.get(c, 1.0) for c in ["Latency (ns)", "Area (mm^2)", "Energy (nJ)", "Leakage (mW)"]]
        t_logs = np.log10(np.clip(t_vals, 1e-12, None))
        weights = [1.0 if c in targets else 0.0 for c in ["Latency (ns)", "Area (mm^2)", "Energy (nJ)", "Leakage (mW)"]]
        return torch.tensor(t_logs, dtype=torch.float32, device=self.device), torch.tensor(weights, dtype=torch.float32, device=self.device)

    def optimize(self, targets, fixed_context, steps=300):
        # Initial architectural guess (power-of-2 normalized log-space)
        # Sequence: [log10_cap, log2_ww, log2_assoc, log2_stack]
        arch_params = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([arch_params], lr=0.02)
        
        # Optimization loop
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Map [0,1] to physical Log-Bounds
            # Cap: 2KB-32MB (log10), WW: 64-2048, Assoc: 1-64, Stack: 1-16
            cap = 10 ** (arch_params[0] * (np.log10(32) - np.log10(2/1024)) + np.log10(2/1024))
            ww  = 2 ** (arch_params[1] * (11 - 6) + 6)
            asc = 2 ** (arch_params[2] * (6 - 0) + 0)
            stk = 2 ** (arch_params[3] * (4 - 0) + 0)
            
            # Build input row
            feat_dict = fixed_context.copy()
            feat_dict.update({"capacity_mb": cap, "word_width": ww, "associativity": asc, "stacked_die_count": stk})
            
            # Create Differentiable Feature Vector
            row = []
            for c in self.feature_cols:
                if c == "capacity_mb": row.append(torch.log10(cap))
                elif c == "word_width": row.append(torch.log2(ww))
                elif c == "associativity": row.append(torch.log2(asc))
                elif c == "stacked_die_count": row.append(torch.log2(stk))
                else: 
                    val = feat_dict.get(c, 0.0)
                    row.append(torch.tensor(val if not isinstance(val, bool) else float(val), device=self.device))
            
            x = (torch.stack(row) - self.means) / self.stds
            pred = self.model(x.unsqueeze(0))
            
            t_tensor, w_tensor = self._get_target_tensor(targets)
            loss = (w_tensor * (pred - t_tensor)**2).sum()
            
            # Boundary penalty
            penalty = torch.relu(-arch_params).sum() + torch.relu(arch_params - 1.0).sum()
            (loss + penalty).backward()
            optimizer.step()
            with torch.no_grad(): arch_params.clamp_(0, 1)

        # Snap result to valid hardware config
        with torch.no_grad():
            final_cap_mb = 10 ** (arch_params[0] * (np.log10(32) - np.log10(2/1024)) + np.log10(2/1024))
            final_cap_kb = 2 ** round(np.log2(final_cap_mb.item() * 1024))
            
            res = {
                "capacity_kb": int(np.clip(final_cap_kb, 2, 32768)),
                "word_width":  int(2 ** round((arch_params[1] * (11 - 6) + 6).item())),
                "associativity": int(2 ** round((arch_params[2] * (6 - 0) + 0).item())),
                "stacked_die_count": int(2 ** round((arch_params[3] * (4 - 0) + 0).item()))
            }
            # Add context for reporting
            res.update(fixed_context)
            
            # Final prediction (unscaled)
            pred_ppa = 10 ** pred.detach().cpu().numpy()[0]
            ppa_dict = {label: pred_ppa[i] for i, label in enumerate(["Latency", "Area", "Energy", "Leakage"])}
            
            return res, ppa_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified DESTINY Inverse Design")
    parser.add_argument("--tech", default="SRAM")
    parser.add_argument("--arch", action="store_true", help="Use architectural-only model")
    parser.add_argument("--target-latency", type=float)
    parser.add_argument("--target-area", type=float)
    parser.add_argument("--target-energy", type=float)
    parser.add_argument("--node", type=int, default=32)
    parser.add_argument("--roadmap", default="HP")
    args = parser.parse_args()

    targets = {}
    if args.target_latency: targets["Latency (ns)"] = args.target_latency
    if args.target_area:    targets["Area (mm^2)"] = args.target_area
    if args.target_energy:  targets["Energy (nJ)"] = args.target_energy

    if not targets:
        print("Error: No targets specified.")
        exit(1)

    # Build context (Physics are defaults for _arch mode)
    context = {"CellInput_ProcessNode": args.node, "DeviceRoadmap_" + args.roadmap: 1.0}
    
    opt = UnifiedOptimizer(args.tech, is_arch=args.arch)
    design, ppa = opt.optimize(targets, context)

    print(f"\n--- Optimized {args.tech} Design (Arch Mode: {args.arch}) ---")
    print(f"Node: {args.node}nm | Roadmap: {args.roadmap}")
    for k, v in design.items():
        if "kb" in k or "width" in k or "assoc" in k or "die" in k:
            print(f"{k:20}: {v}")
    
    print("\n--- Predicted Performance vs Targets ---")
    for k, v in ppa.items():
        target_val = targets.get(f"{k} ({'ns' if k=='Latency' else 'mm^2' if k=='Area' else 'nJ' if k=='Energy' else 'mW'})", "-")
        print(f"{k:10}: {v:10.4f}  (Target: {target_val})")

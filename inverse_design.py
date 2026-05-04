#!/usr/bin/env python3
#### TODO: THIS WILL NEED TO BE EDITED. IT CURRENTLY ONLY CONTAINS THE FEATURES USED FOR THE ARCH SWEEP
import os, argparse, json, pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from train_model import PPA_MLP
# TODO: import from train_model import FeasibilityClassifier
import sys

# Target column order must match the 4-output PPA model
TARGET_KEYS   = ["cache_hit_latency_ns", "cache_write_energy_nJ", "cache_area_mm2", "cache_leakage_mW"]
TARGET_LABELS = ["Latency", "Energy", "Area", "Leakage"]

# Architectural parameters: [log10_cap, log2_ww, log2_assoc, log2_stack]
# Bounds: Cap 2KB–32MB, WW 64–2048 (2^6–2^11), Assoc 1–64 (2^0–2^6), Stack 1–16 (2^0–2^4)
ARCH_BOUNDS = [
    (np.log10(2),  np.log10(32768)),  # capacity_kb  (log10)
    (6,            11),               # word_width    (log2)
    (0,            6),                # associativity (log2)
    (0,            4),                # stacked_die   (log2)
]
ARCH_COLS = ["capacity_kb", "word_width_bits", "associativity", "data_stacked_die_count"]


class InverseOptimizer:
    def __init__(self, tech, is_arch=False):
        self.tech    = tech
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        suffix       = "_arch_full" if is_arch else "_full"
        model_dir    = f"model_output/{tech.lower()}{suffix}"

        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        # Infer architecture from saved state dict
        sd        = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        hidden    = sd["input_proj.weight"].shape[0]
        n_blocks  = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1

        self.model = PPA_MLP(len(self.feature_cols), hidden_dim=hidden, n_blocks=n_blocks).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        # TODO: Load second model (feasibility classifier) weights here

        self.means = torch.tensor(self.scaler.mean_,  dtype=torch.float32, device=self.device)
        self.stds  = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

    def _target_tensors(self, targets):
        """Return (log10 target values, binary weight mask) for the 4 PPA outputs."""
        vals    = [targets.get(k, 1.0) for k in TARGET_KEYS]
        weights = [1.0 if k in targets else 0.0 for k in TARGET_KEYS]
        return (
            torch.tensor(np.log10(np.clip(vals, 1e-12, None)), dtype=torch.float32, device=self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
        )

    def optimize(self, targets, fixed_context, steps=300):
        t_tensor, w_tensor = self._target_tensors(targets)

        # Indices of dynamically updated columns in the feature vector
        dyn_keys = ARCH_COLS + ["derived_sqrt_capacity", "derived_cap_per_die", "derived_rows_per_die"] + \
                   [f"device_roadmap_{rm}_x_log10_cap" for rm in ["HP", "LOP", "LSTP"]]
        dyn_idx  = {k: i for i, k in enumerate(self.feature_cols) if k in dyn_keys}

        # Static base vector — filled once from fixed_context
        base_x = torch.zeros(len(self.feature_cols), device=self.device)
        for i, c in enumerate(self.feature_cols):
            if c not in dyn_idx:
                base_x[i] = float(fixed_context.get(c, 0.0))

        # Architectural params in [0, 1] — mapped to physical log-space during optimisation
        arch = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True, device=self.device)
        opt  = torch.optim.Adam([arch], lr=0.02)

        for _ in range(steps):
            opt.zero_grad()

            # Map [0,1] -> physical log-space using per-param bounds
            lo, hi   = zip(*ARCH_BOUNDS)
            lo, hi   = torch.tensor(lo, device=self.device), torch.tensor(hi, device=self.device)
            log_vals = arch * (hi - lo) + lo   # [cap_log10, ww_log2, assoc_log2, stk_log2]
            cap_phys = 10 ** log_vals[0]

            # Build differentiable feature vector
            x = base_x.clone()
            for col, lv in zip(ARCH_COLS, log_vals):
                if col in dyn_idx: x[dyn_idx[col]] = lv
            if "derived_sqrt_capacity" in dyn_idx:
                x[dyn_idx["derived_sqrt_capacity"]] = torch.sqrt(cap_phys)
            # die_count in physical space (2^log2 value)
            stk_phys = 2 ** log_vals[3]
            ww_phys  = 2 ** log_vals[1]
            if "derived_cap_per_die" in dyn_idx:
                x[dyn_idx["derived_cap_per_die"]]  = cap_phys / stk_phys
            if "derived_rows_per_die" in dyn_idx:
                x[dyn_idx["derived_rows_per_die"]] = (cap_phys * 1024) / (ww_phys * stk_phys)
            for rm in ["HP", "LOP", "LSTP"]:
                col = f"device_roadmap_{rm}_x_log10_cap"
                if col in dyn_idx:
                    x[dyn_idx[col]] = float(fixed_context.get(f"device_roadmap_{rm}", 0.0)) * log_vals[0]

            pred = self.model(((x - self.means) / self.stds).unsqueeze(0))
            loss = (w_tensor * (pred - t_tensor) ** 2).sum()
            
            # TODO: Add second model (feasibility classifier) penalty here
            # feasibility_loss = self.classifier_model(x_scaled)
            # loss += feasibility_loss

            # Penalty keeps arch params in [0, 1]
            penalty = torch.relu(-arch).sum() + torch.relu(arch - 1.0).sum()
            (loss + penalty).backward()
            opt.step()
            with torch.no_grad(): arch.clamp_(0, 1)

        # Snap to nearest valid power-of-2 hardware config
        with torch.no_grad():
            lo, hi   = zip(*ARCH_BOUNDS)
            log_vals = arch * (torch.tensor(hi, device=self.device) - torch.tensor(lo, device=self.device)) \
                       + torch.tensor(lo, device=self.device)

            cap_kb = 2 ** round(np.log2(10 ** log_vals[0].item()))
            design = {
                "capacity_kb":            int(np.clip(cap_kb, 2, 32768)),
                "word_width_bits":        int(2 ** round(log_vals[1].item())),
                "associativity":          int(2 ** round(log_vals[2].item())),
                "data_stacked_die_count": int(2 ** round(log_vals[3].item())),
            }
            design.update(fixed_context)

            pred_ppa = 10 ** pred.detach().cpu().numpy()[0]
            ppa_dict = {label: pred_ppa[i] for i, label in enumerate(TARGET_LABELS)}

        return design, ppa_dict


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DESTINY Inverse Design Optimizer")
    p.add_argument("--tech",             default="SRAM")
    p.add_argument("--arch",             action="store_true",  help="Use architectural-only model")
    p.add_argument("--target-latency",   type=float,           help="Target read latency (ns)")
    p.add_argument("--target-area",      type=float,           help="Target cache area (mm²)")
    p.add_argument("--target-energy",    type=float,           help="Target hit energy (nJ)")
    p.add_argument("--target-leakage",   type=float,           help="Target leakage power (mW)")
    p.add_argument("--node",             type=int,   default=32, choices=[22, 32, 45, 65])
    p.add_argument("--roadmap",          default="HP",           choices=["HP", "LOP", "LSTP"])
    p.add_argument("--temperature",      type=float, default=350.0, help="Temperature (K)")
    p.add_argument("--output",           default=None,           help="CSV to append results to")
    args = p.parse_args()

    targets = {k: v for k, v in [
        ("cache_hit_latency_ns", args.target_latency),
        ("cache_area_mm2",       args.target_area),
        ("cache_write_energy_nJ",  args.target_energy),
        ("cache_leakage_mW",     args.target_leakage),
    ] if v is not None}

    if not targets:
        p.error("Specify at least one target: --target-latency / --target-area / --target-energy / --target-leakage")

    context = {
        f"process_node_nm_{args.node}":  1.0,
        f"device_roadmap_{args.roadmap}": 1.0,
        "temperature_K":                  args.temperature,
    }

    design, ppa = InverseOptimizer(args.tech, is_arch=args.arch).optimize(targets, context)

    row = {
        "tech": args.tech, "arch": args.arch, "node_nm": args.node,
        "roadmap": args.roadmap, "temperature_K": args.temperature,
        # Optimized design
        "capacity_kb":            design.get("capacity_kb"),
        "word_width_bits":        design.get("word_width_bits"),
        "associativity":          design.get("associativity"),
        "data_stacked_die_count": design.get("data_stacked_die_count"),
        # Surrogate-predicted PPA
        "pred_latency_ns":  ppa.get("Latency"),
        "pred_area_mm2":    ppa.get("Area"),
        "pred_energy_nJ":   ppa.get("Energy"),
        "pred_leakage_mW":  ppa.get("Leakage"),
        # Requested targets (NaN when not specified)
        "target_latency_ns":  targets.get("cache_hit_latency_ns", float("nan")),
        "target_area_mm2":    targets.get("cache_area_mm2",        float("nan")),
        "target_energy_nJ":   targets.get("cache_write_energy_nJ",  float("nan")),
        "target_leakage_mW":  targets.get("cache_leakage_mW",     float("nan")),
    }

    df_out = pd.DataFrame([row])
    if args.output:
        df_out.to_csv(args.output, mode="a", index=False, header=not os.path.exists(args.output))
        print(f"Result appended to {args.output}", file=sys.stderr)
    else:
        print(df_out.to_csv(index=False), end="")
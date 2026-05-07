#!/usr/bin/env python3
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

# Architectural parameters: [log10_cap, log2_ww, log2_assoc, log2_stack, ...]
# Basic bounds: Cap 2KB–32MB, WW 64–2048 (2^6–2^11), Assoc 1–64 (2^0–2^6), Stack 1–16 (2^0–2^4)
BASE_ARCH_BOUNDS = [
    (np.log10(2),  np.log10(32768)),  # capacity_kb  (log10)
    (6,            11),               # word_width    (log2)
    (0,            6),                # associativity (log2)
    (0,            4),                # stacked_die   (log2)
]
BASE_ARCH_COLS = ["capacity_kb", "word_width_bits", "associativity", "data_stacked_die_count"]

# Organizational 'data_' parameters that can be optionally included in optimization
DATA_PARAM_BOUNDS_LOG2 = {
    "data_mux_sense_amp":               (0, 8),   # 1–256
    "data_mux_output_lev2":             (0, 8),   # 1–256
    "data_num_active_mat_per_row":      (0, 9),   # 1–512
    "data_num_active_mat_per_col":      (0, 9),   # 1–512
    "data_num_row_per_set":             (0, 8),   # 1–256
}

DATA_PARAM_BOUNDS_LINEAR = {
    "data_num_active_subarray_per_row": (1, 2),
    "data_num_active_subarray_per_col": (1, 2),
}

SRAM_CELL_PARAM_BOUNDS_LINEAR = {
    "CellInput_SRAMCellNMOSWidth (F)": (1.0, 2.5),
    "CellInput_SRAMCellPMOSWidth (F)": (1.0, 2.5),
    "CellInput_AccessCMOSWidth (F)":   (1.0, 2.5),
    "CellInput_ReadVoltage (V)":        (0.5, 1.2),
}

class InverseOptimizer:
    def __init__(self, tech, is_arch=False):
        self.tech    = tech
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prioritize the model with data params if it exists, otherwise fall back to standard
        suffix_with_data = "_arch_full_with_data_params" if is_arch else "_full_with_data_params"
        suffix_standard  = "_arch_full" if is_arch else "_full"
        
        model_dir = f"model_output/{tech.lower()}{suffix_with_data}"
        if not os.path.exists(model_dir):
            model_dir = f"model_output/{tech.lower()}{suffix_standard}"

        if not os.path.exists(model_dir):
            print(f"WARNING: Model directory {model_dir} not found, trying default 'model_output'")
            model_dir = "model_output"

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

        self.means = torch.tensor(self.scaler.mean_,  dtype=torch.float32, device=self.device)
        self.stds  = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

        # Identify which columns in feature_cols are architectural/organizational inputs
        self.opt_cols    = []
        self.opt_bounds  = []
        self.log10_cols  = []
        self.log2_cols   = []
        self.linear_cols = []
        
        # Add base architectural columns
        for col, bound in zip(BASE_ARCH_COLS, BASE_ARCH_BOUNDS):
            if col in self.feature_cols:
                self.opt_cols.append(col)
                self.opt_bounds.append(bound)
                if col == "capacity_kb":
                    self.log10_cols.append(col)
                else:
                    self.log2_cols.append(col)
        
        # Add data organizational columns (log2 and linear)
        for col, bound in DATA_PARAM_BOUNDS_LOG2.items():
            if col in self.feature_cols:
                self.opt_cols.append(col)
                self.opt_bounds.append(bound)
                self.log2_cols.append(col)

        for col, bound in DATA_PARAM_BOUNDS_LINEAR.items():
            if col in self.feature_cols:
                self.opt_cols.append(col)
                self.opt_bounds.append(bound)
                self.linear_cols.append(col)

        # Add SRAM cell parameters if tech is SRAM
        if self.tech == "SRAM":
            for col, bound in SRAM_CELL_PARAM_BOUNDS_LINEAR.items():
                if col in self.feature_cols:
                    self.opt_cols.append(col)
                    self.opt_bounds.append(bound)
                    self.linear_cols.append(col)

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
        dyn_keys = self.opt_cols + ["derived_sqrt_capacity", "derived_cap_per_die", "derived_rows_per_die",
                                    "CellInput_CellArea (F^2)", "derived_sqrt_area", "derived_read_v_sq"] + \
                   [f"device_roadmap_{rm}_x_log10_cap" for rm in ["HP", "LOP", "LSTP"]]
        dyn_idx  = {k: i for i, k in enumerate(self.feature_cols) if k in dyn_keys}

        # Static base vector — filled once from fixed_context
        base_x = torch.zeros(len(self.feature_cols), device=self.device)
        for i, c in enumerate(self.feature_cols):
            if c not in dyn_idx:
                base_x[i] = float(fixed_context.get(c, 0.0))

        # Parameters in [0, 1] — mapped to physical log-space during optimisation
        n_params = len(self.opt_cols)
        params = torch.full((n_params,), 0.5, requires_grad=True, device=self.device)
        opt    = torch.optim.Adam([params], lr=0.02)

        lo_bound, hi_bound = zip(*self.opt_bounds)
        lo_bound = torch.tensor(lo_bound, device=self.device)
        hi_bound = torch.tensor(hi_bound, device=self.device)

        for _ in range(steps):
            opt.zero_grad()

            # Map [0,1] -> physical log-space
            log_vals = params * (hi_bound - lo_bound) + lo_bound
            
            # Extract capacity (log10) for derived features
            cap_log10 = log_vals[self.opt_cols.index("capacity_kb")]
            cap_phys  = 10 ** cap_log10

            # Build differentiable feature vector
            x = base_x.clone()
            for col, lv in zip(self.opt_cols, log_vals):
                if col in dyn_idx: x[dyn_idx[col]] = lv
            
            if "derived_sqrt_capacity" in dyn_idx:
                x[dyn_idx["derived_sqrt_capacity"]] = torch.sqrt(cap_phys)
            
            # die_count, ww in physical space for derived ratios
            stk_phys = 2 ** log_vals[self.opt_cols.index("data_stacked_die_count")]
            ww_phys  = 2 ** log_vals[self.opt_cols.index("word_width_bits")]
            
            if "derived_cap_per_die" in dyn_idx:
                x[dyn_idx["derived_cap_per_die"]]  = cap_phys / stk_phys
            if "derived_rows_per_die" in dyn_idx:
                x[dyn_idx["derived_rows_per_die"]] = (cap_phys * 1024) / (ww_phys * stk_phys)
            
            # SRAM cell parameter optimization: Area and derived features computed from widths
            if self.tech == "SRAM" and "CellInput_SRAMCellNMOSWidth (F)" in self.opt_cols:
                wn  = log_vals[self.opt_cols.index("CellInput_SRAMCellNMOSWidth (F)")]
                wp  = log_vals[self.opt_cols.index("CellInput_SRAMCellPMOSWidth (F)")]
                wac = log_vals[self.opt_cols.index("CellInput_AccessCMOSWidth (F)")]
                rv  = log_vals[self.opt_cols.index("CellInput_ReadVoltage (V)")]

                cell_area = 60 + 20*(wn + wac) + 10*wp
                if "CellInput_CellArea (F^2)" in dyn_idx:
                    x[dyn_idx["CellInput_CellArea (F^2)"]] = torch.log10(cell_area)
                if "derived_sqrt_area" in dyn_idx:
                    x[dyn_idx["derived_sqrt_area"]] = torch.sqrt(cell_area)
                if "derived_read_v_sq" in dyn_idx:
                    x[dyn_idx["derived_read_v_sq"]] = rv ** 2
            
            for rm in ["HP", "LOP", "LSTP"]:
                col = f"device_roadmap_{rm}_x_log10_cap"
                if col in dyn_idx:
                    x[dyn_idx[col]] = float(fixed_context.get(f"device_roadmap_{rm}", 0.0)) * cap_log10

            pred = self.model(((x - self.means) / self.stds).unsqueeze(0))
            loss = (w_tensor * (pred - t_tensor) ** 2).sum()
            
            # Penalty keeps params in [0, 1]
            penalty = torch.relu(-params).sum() + torch.relu(params - 1.0).sum()
            (loss + penalty).backward()
            opt.step()
            with torch.no_grad(): params.clamp_(0, 1)

        # Snap to valid hardware config
        with torch.no_grad():
            final_log_vals = params * (hi_bound - lo_bound) + lo_bound
            
            design = {}
            for col, lv in zip(self.opt_cols, final_log_vals.cpu().numpy()):
                if col in self.log10_cols:
                    if col == "capacity_kb":
                        # Snap capacity to nearest power of 2
                        val = 2 ** round(np.log2(10 ** lv))
                        design[col] = int(np.clip(val, 2, 32768))
                    else:
                        design[col] = 10 ** lv
                elif col in self.log2_cols:
                    # Log2 snap: 2^round(lv)
                    design[col] = int(2 ** round(lv))
                elif col in self.linear_cols:
                    if col in SRAM_CELL_PARAM_BOUNDS_LINEAR:
                        # Widths and ReadVoltage snap to 2 decimal places
                        design[col] = float(round(lv, 2))
                    else:
                        # Other linear (e.g. subarray count) snaps to nearest integer
                        design[col] = int(round(lv))
            
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

    # Technology context + optimized architectural/cell design
    row = {
        "tech": args.tech, "arch": args.arch, "node_nm": args.node,
        "roadmap": args.roadmap, "temperature_K": args.temperature,
    }
    row.update(design)

    # Surrogate-predicted PPA
    row.update({
        "pred_latency_ns":  ppa.get("Latency"),
        "pred_area_mm2":    ppa.get("Area"),
        "pred_energy_nJ":   ppa.get("Energy"),
        "pred_leakage_mW":  ppa.get("Leakage"),
    })

    # Requested targets (NaN when not specified)
    row.update({
        "target_latency_ns":  targets.get("cache_hit_latency_ns", float("nan")),
        "target_area_mm2":    targets.get("cache_area_mm2",        float("nan")),
        "target_energy_nJ":   targets.get("cache_write_energy_nJ",  float("nan")),
        "target_leakage_mW":  targets.get("cache_leakage_mW",     float("nan")),
    })

    df_out = pd.DataFrame([row])
    if args.output:
        df_out.to_csv(args.output, mode="a", index=False, header=not os.path.exists(args.output))
        print(f"Result appended to {args.output}", file=sys.stderr)
    else:
        print(df_out.to_csv(index=False), end="")
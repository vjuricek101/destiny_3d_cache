#!/usr/bin/env python3
import os, argparse, json, pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from train_model import PPA_MLP
import sys
from destiny_utils import (
    TARGET_COLS as TARGET_KEYS,
    TARGET_LABELS,
    TARGET_SHORT_LABELS,
    get_active_targets,
)

# Architectural bounds (log10 for capacity, log2 for everything else)
BASE_ARCH_BOUNDS = [
    (np.log10(2),  np.log10(32768)),  # capacity_kb  (log10)
    (6,            11),               # word_width    (log2)
    (0,            6),                # associativity (log2)
    (0,            4),                # stacked_die   (log2)
]
BASE_ARCH_COLS = ["capacity_kb", "word_width_bits", "associativity", "data_stacked_die_count"]

DATA_PARAM_BOUNDS_LOG2 = {
    "data_mux_sense_amp":           (0, 6),
    "data_mux_output_lev1":         (0, 6),
    "data_mux_output_lev2":         (0, 6),
    "data_num_active_mat_per_row":  (0, 4),
    "data_num_active_mat_per_col":  (0, 4),
    "tag_num_row_mat":              (0, 6),
    "tag_num_col_mat":              (0, 6),
    "tag_mux_sense_amp":            (0, 6),
    "tag_mux_output_lev1":          (0, 6),
    "tag_mux_output_lev2":          (0, 6),
    "tag_num_active_mat_per_row":   (0, 6),
    "tag_num_active_mat_per_col":   (0, 6),
}
DATA_PARAM_BOUNDS_LINEAR = {
    "data_num_active_subarray_per_row": (1, 2),
    "data_num_active_subarray_per_col": (1, 2),
    "tag_num_active_subarray_per_row":  (1, 2),
    "tag_num_active_subarray_per_col":  (1, 2),
}

_MUX_VALID = [1, 2, 4, 8, 16, 32, 64]
_MAT_VALID = [1, 2, 4, 8, 16]

def _nearest_valid(val: int, valid_set: list) -> int:
    return min(valid_set, key=lambda v: abs(v - val))

SRAM_CELL_BOUNDS_LOG10 = {
    "CellInput_SRAMCellNMOSWidth (F)": (2.2, 2.5),
    "CellInput_SRAMCellPMOSWidth (F)": (1.0, 1.1),
    "CellInput_AccessCMOSWidth (F)":   (1.1, 1.25),
}
SRAM_CELL_BOUNDS_LINEAR = {
    "CellInput_ReadVoltage (V)": (0.5, 1.2),
}

# -- Module-level helpers ------------------------------------------------------

def _build_dyn_idx(feature_cols, opt_cols):
    """Index of all columns updated per gradient step."""
    dyn_keys = set(opt_cols) | {
        "derived_sqrt_capacity", "derived_cap_per_die", "derived_rows_per_die",
        "CellInput_CellArea (F^2)", "derived_sqrt_area", "derived_read_v_sq",
        "CellInput_MinSenseVoltage (mV)", "CellInput_CellAspectRatio",
    } | {f"device_roadmap_{rm}_x_log10_cap" for rm in ["HP", "LOP", "LSTP"]}
    return {k: i for i, k in enumerate(feature_cols) if k in dyn_keys}


def _update_sram_cell_features(x, dyn_idx, log_vals, opt_cols, fixed_context, device):
    """Compute SRAM cell derived features and write them into feature vector x (in-place).

    CellArea is computed using differentiable torch ops so gradients flow back to
    wn/wp/wac during optimization. 
    """
    wn  = 10 ** log_vals[opt_cols.index("CellInput_SRAMCellNMOSWidth (F)")]
    wp  = 10 ** log_vals[opt_cols.index("CellInput_SRAMCellPMOSWidth (F)")]
    wac = 10 ** log_vals[opt_cols.index("CellInput_AccessCMOSWidth (F)")]
    if "CellInput_ReadVoltage (V)" in opt_cols:
        rv = log_vals[opt_cols.index("CellInput_ReadVoltage (V)")]
    else:
        rv = torch.tensor(float(fixed_context.get("CellInput_ReadVoltage (V)", 1.0)), device=device)

    # Differentiable CellArea — gradient flows back to wn, wp, wac
    cell_area = 55.0 + 30.0 * torch.maximum(wn, wac) + 20.0 * (wp + 0.5) # Mirrors the formula in derive_sram_physical_params
    cell_area = torch.clamp(cell_area, 40.0, 200.0) # Same clamping too

    if "CellInput_CellArea (F^2)" in dyn_idx:
        x[dyn_idx["CellInput_CellArea (F^2)"]] = torch.log10(cell_area)
    if "derived_sqrt_area" in dyn_idx:
        x[dyn_idx["derived_sqrt_area"]] = torch.sqrt(cell_area)
    if "derived_read_v_sq" in dyn_idx:
        x[dyn_idx["derived_read_v_sq"]] = rv ** 2

    # Dynamically update MinSenseVoltage and AspectRatio so the model does not see out-of-distribution 0.0 values
    if "CellInput_MinSenseVoltage (mV)" in dyn_idx:
        a_vth = 3.0
        for k in fixed_context:
            if k.startswith("process_node_nm_") and float(fixed_context[k]) > 0.5:
                node = int(k.split("_")[-1])
                a_vth = {65: 5.0, 45: 4.0, 32: 3.0, 22: 2.5}.get(node, 3.0)
                break
        v_sense = 6.0 * a_vth / torch.sqrt(2.0 * wac)
        v_sense = torch.clamp(v_sense, 5.0, 80.0)
        x[dyn_idx["CellInput_MinSenseVoltage (mV)"]] = v_sense

    if "CellInput_CellAspectRatio" in dyn_idx:
        x[dyn_idx["CellInput_CellAspectRatio"]] = torch.tensor(1.4600, device=device)


def _snap_design(opt_cols, final_log_vals, log10_cols, log2_cols, linear_cols):
    """Convert continuous log-space optimised params to snapped physical values."""
    design = {}
    for col, lv in zip(opt_cols, final_log_vals):
        if col in log10_cols:
            if col == "capacity_kb":
                val = 2 ** round(np.log2(10 ** lv))
                design[col] = int(np.clip(val, 2, 32768))
            else:
                design[col] = 10 ** lv
        elif col in log2_cols:
            raw = int(2 ** round(lv))
            if "mux_" in col:
                design[col] = _nearest_valid(raw, _MUX_VALID)
            elif "_mat_" in col or "num_row_mat" in col or "num_col_mat" in col:
                design[col] = _nearest_valid(raw, _MAT_VALID)
            else:
                design[col] = raw
        elif col in linear_cols:
            design[col] = int(round(lv)) if col in DATA_PARAM_BOUNDS_LINEAR else float(round(lv, 2))
    return design


def _rows_per_subarray_penalty(log_vals, opt_cols, cap_log10, ww_log2):
    """Differentiable physics penalty enforcing the data-array rows-per-subarray identity.

    DESTINY requires (BankWithoutHtree.cpp:113, BankWithHtree.cpp:381):
        rows_per_subarray = (cap_bytes * 8) / (assoc * wordWidth * numActiveMatPerCol * numActiveSubarrayPerCol)

    In log2-space this is:
        log2_rps = log2(cap_bits) - log2(assoc) - log2(ww) - log2(nmac_col) - log2(nsac_col)

    Two soft sub-penalties are summed (caller applies a scalar weight):
        relu(-log2_rps)                               -- negativity: not enough rows
        (log2_rps - round(log2_rps).detach())^2      -- divisibility: non-integer log2

    Parameters
    ----------
    log_vals  : 1-D Tensor of mapped optimizer values (log10/log2/linear per column type)
    opt_cols  : list[str] in the same order as log_vals
    cap_log10 : scalar Tensor, log10(capacity_kb)
    ww_log2   : scalar Tensor, log2(word_width_bits)
    """
    # cap_bits = cap_kb * 8192 = cap_kb * 2^13
    # log2(cap_bits) = log2(cap_kb) + 13 = cap_log10 * log2(10) + 13
    LOG2_10 = 3.321928094887362
    log2_cap_bits = cap_log10 * LOG2_10 + 13.0

    def _get_log2(col, default_log2=0.0):
        """Return the log2-space entry for a log2-encoded column, else a scalar constant."""
        if col in opt_cols:
            return log_vals[opt_cols.index(col)]
        # Produce a zero-gradient constant on the same device as cap_log10
        return cap_log10.new_zeros(1).squeeze() + default_log2

    log2_assoc   = _get_log2("associativity", 2.0)           # default: log2(4)
    log2_nac_col = _get_log2("data_num_active_mat_per_col", 0.0)  # log2 space, default: log2(1)

    # data_num_active_subarray_per_col lives in LINEAR space [1, 2]; convert to log2
    if "data_num_active_subarray_per_col" in opt_cols:
        nsac_lin  = log_vals[opt_cols.index("data_num_active_subarray_per_col")]
        log2_nsac = torch.log2(nsac_lin.clamp(min=1.0))
    else:
        log2_nsac = cap_log10.new_zeros(1).squeeze()  # log2(1) = 0

    log2_rps = log2_cap_bits - log2_assoc - ww_log2 - log2_nac_col - log2_nsac

    # Penalty 1: rows_per_subarray must be >= 1
    neg_penalty = torch.relu(-log2_rps)

    # Penalty 2: log2(rows) must be a non-negative integer (power of two)
    # Detach the rounded value so gradient flows through log2_rps, not through round()
    frac = log2_rps - torch.round(log2_rps.detach())
    int_penalty = frac ** 2

    return neg_penalty + int_penalty


# -- Optimizer -----------------------------------------------------------------

class InverseOptimizer:
    def __init__(self, tech):
        self.tech   = tech
        self.device = torch.device("cpu")

        model_dir = f"model_output/{tech.lower()}_feasibility"
        if not os.path.exists(model_dir):
            print(f"WARNING: {model_dir} not found, trying default 'model_output'")
            model_dir = "model_output"

        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        sd       = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        hidden   = sd["input_proj.weight"].shape[0]
        n_blocks = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
        n_targets = sd["output_head.weight"].shape[0]   # e.g. 6 for SRAM, 8 for eDRAM
        self.model = PPA_MLP(len(self.feature_cols), hidden_dim=hidden,
                             n_blocks=n_blocks, n_targets=n_targets).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        # Length of the model output head and the target/weight tensors
        self._active_keys = get_active_targets(self.tech)

        self.means = torch.tensor(self.scaler.mean_,  dtype=torch.float32, device=self.device)
        self.stds  = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

        # Candidate columns — pruned per-call in optimize() based on fixed_context
        self._all_opt_cols    = []
        self._all_opt_bounds  = []
        self._all_log10_cols  = []
        self._all_log2_cols   = []
        self._all_linear_cols = []

        for col, bound in zip(BASE_ARCH_COLS, BASE_ARCH_BOUNDS):
            if col in self.feature_cols:
                self._all_opt_cols.append(col); self._all_opt_bounds.append(bound)
                (self._all_log10_cols if col == "capacity_kb" else self._all_log2_cols).append(col)

        for col, bound in DATA_PARAM_BOUNDS_LOG2.items():
            if col in self.feature_cols:
                self._all_opt_cols.append(col); self._all_opt_bounds.append(bound)
                self._all_log2_cols.append(col)

        for col, bound in DATA_PARAM_BOUNDS_LINEAR.items():
            if col in self.feature_cols:
                self._all_opt_cols.append(col); self._all_opt_bounds.append(bound)
                self._all_linear_cols.append(col)

        if self.tech == "SRAM":
            for col, bound in SRAM_CELL_BOUNDS_LOG10.items():
                if col in self.feature_cols:
                    self._all_opt_cols.append(col)
                    self._all_opt_bounds.append((np.log10(bound[0]), np.log10(bound[1])))
                    self._all_log10_cols.append(col)
            for col, bound in SRAM_CELL_BOUNDS_LINEAR.items():
                if col in self.feature_cols:
                    self._all_opt_cols.append(col); self._all_opt_bounds.append(bound)
                    self._all_linear_cols.append(col)

    def _target_tensors(self, targets):
        """Return (log10 target values, binary weight mask) aligned to the model output head.

        Uses self._active_keys — the same ordered subset of TARGET_KEYS the model was
        trained on — so tensor length matches the checkpoint's output_head shape exactly.
        """
        vals    = [targets.get(k, 1.0) for k in self._active_keys]
        weights = [1.0 if k in targets else 0.0 for k in self._active_keys]
        return (
            torch.tensor(np.log10(np.clip(vals, 1e-12, None)), dtype=torch.float32, device=self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
        )

    def optimize(self, targets, fixed_context, steps=300, n_restarts=4, target_weights=None, verbose=False):
        """Gradient-based inverse design with multi-start."""
        # Exclude columns that are pinned by fixed_context so the optimizer
        # cannot move them during gradient descent.
        fixed_keys = set(fixed_context.keys())
        self.opt_cols    = [c for c in self._all_opt_cols    if c not in fixed_keys]
        self.opt_bounds  = [b for c, b in zip(self._all_opt_cols, self._all_opt_bounds) if c not in fixed_keys]
        self.log10_cols  = [c for c in self._all_log10_cols  if c not in fixed_keys]
        self.log2_cols   = [c for c in self._all_log2_cols   if c not in fixed_keys]
        self.linear_cols = [c for c in self._all_linear_cols if c not in fixed_keys]

        t_tensor, w_tensor = self._target_tensors(targets)
        if target_weights is not None:
            if isinstance(target_weights, dict):
                w_tensor = torch.tensor(
                    [target_weights.get(k, 1.0 if k in targets else 0.0) for k in TARGET_KEYS],
                    dtype=torch.float32, device=self.device)
            else:
                w_tensor = torch.tensor(target_weights, dtype=torch.float32, device=self.device)

        dyn_idx = _build_dyn_idx(self.feature_cols, self.opt_cols)

        # Static base vector from fixed_context (dynamic cols filled per step)
        base_x = torch.zeros(len(self.feature_cols), device=self.device)
        for i, c in enumerate(self.feature_cols):
            if c not in dyn_idx:
                base_x[i] = float(fixed_context.get(c, 0.0))



        lo_bound = torch.tensor([b[0] for b in self.opt_bounds], device=self.device)
        hi_bound = torch.tensor([b[1] for b in self.opt_bounds], device=self.device)

        best_loss, best_params, best_pred = float("inf"), None, None

        for restart in range(max(1, n_restarts)):
            params = (torch.full((len(self.opt_cols),), 0.5, device=self.device)
                      if restart == 0 else torch.rand(len(self.opt_cols), device=self.device))
            params = params.requires_grad_(True)
            inner_opt = torch.optim.Adam([params], lr=0.02)

            for _ in range(steps):
                inner_opt.zero_grad()
                log_vals  = params * (hi_bound - lo_bound) + lo_bound
                cap_log10 = log_vals[self.opt_cols.index("capacity_kb")]
                cap_phys  = 10 ** cap_log10

                x = base_x.clone()
                for col, lv in zip(self.opt_cols, log_vals):
                    if col in dyn_idx: x[dyn_idx[col]] = lv

                if "derived_sqrt_capacity" in dyn_idx:
                    x[dyn_idx["derived_sqrt_capacity"]] = torch.sqrt(cap_phys)

                if "data_stacked_die_count" in self.opt_cols:
                    stk_phys = 2 ** log_vals[self.opt_cols.index("data_stacked_die_count")]
                else:
                    stk_phys = torch.tensor(float(fixed_context.get("data_stacked_die_count", 1.0)), device=self.device)

                if "word_width_bits" in self.opt_cols:
                    ww_log2 = log_vals[self.opt_cols.index("word_width_bits")]
                    ww_phys = 2 ** ww_log2
                else:
                    # word_width_bits pinned in fixed_context as its log2 value
                    ww_log2 = torch.tensor(float(fixed_context.get("word_width_bits", 6.0)), device=self.device)
                    ww_phys = 2 ** ww_log2

                if "derived_cap_per_die"  in dyn_idx:
                    x[dyn_idx["derived_cap_per_die"]]  = cap_phys / stk_phys
                if "derived_rows_per_die" in dyn_idx:
                    x[dyn_idx["derived_rows_per_die"]] = (cap_phys * 1024) / (ww_phys * stk_phys)

                if self.tech == "SRAM" and "CellInput_SRAMCellNMOSWidth (F)" in self.opt_cols:
                    _update_sram_cell_features(x, dyn_idx, log_vals, self.opt_cols, fixed_context, self.device)

                for rm in ["HP", "LOP", "LSTP"]:
                    col = f"device_roadmap_{rm}_x_log10_cap"
                    if col in dyn_idx:
                        x[dyn_idx[col]] = float(fixed_context.get(f"device_roadmap_{rm}", 0.0)) * cap_log10

                x_scaled = ((x - self.means) / self.stds).unsqueeze(0)
                pred, p_feas    = self.model.forward_with_feasibility(x_scaled)
                learned_penalty = 50.0 * (1.0 - p_feas.squeeze())
                # Physics: penalise designs where rows_per_subarray is not a positive power-of-two
                physics_penalty = 5.0 * _rows_per_subarray_penalty(log_vals, self.opt_cols, cap_log10, ww_log2)

                loss    = (w_tensor * (pred - t_tensor) ** 2).sum() + learned_penalty + physics_penalty
                barrier = (torch.relu(-params) ** 2).sum() + (torch.relu(params - 1.0) ** 2).sum()
                (loss + 100.0 * barrier).backward()

                with torch.no_grad():
                    params.grad[params <= 0] = params.grad[params <= 0].clamp(max=0)
                    params.grad[params >= 1] = params.grad[params >= 1].clamp(min=0)
                inner_opt.step()
                with torch.no_grad(): params.data.clamp_(0, 1)

            with torch.no_grad():
                final_loss = loss.item()
            if final_loss < best_loss:
                best_loss, best_params, best_pred = final_loss, params.detach().clone(), pred.detach()

        params, pred = best_params, best_pred

        with torch.no_grad():
            final_log_vals = (params * (hi_bound - lo_bound) + lo_bound).cpu().numpy()
            design = _snap_design(self.opt_cols, final_log_vals,
                                  self.log10_cols, self.log2_cols, self.linear_cols)

            # Associativity was zero-variance in training sweep; always default to 4
            if "associativity" not in design:
                design["associativity"] = 4

            # Enforce generate_cells.py SRAM constraints: gamma (wp<wac), beta (wn>=2*wac)
            if self.tech == "SRAM" and "CellInput_SRAMCellNMOSWidth (F)" in design:
                wn  = design["CellInput_SRAMCellNMOSWidth (F)"]
                wp  = design["CellInput_SRAMCellPMOSWidth (F)"]
                wac = design["CellInput_AccessCMOSWidth (F)"]
                if wp / wac >= 1.0:
                    if 0.9 * wac > 1.0:
                        wp = float(round(0.9 * wac, 2))
                    else:
                        wac = float(round(min(1.0 / 0.9 + 0.01, 1.25), 2))
                        wp  = float(round(min(0.9 * wac, 1.1), 2))
                    design["CellInput_AccessCMOSWidth (F)"]   = wac
                    design["CellInput_SRAMCellPMOSWidth (F)"] = wp
                if wn / wac < 2.0:
                    design["CellInput_SRAMCellNMOSWidth (F)"] = float(round(min(2.0 * wac, 2.5), 2))

            # Values already decoded to physical values in fixed_context
            for k, v in fixed_context.items():
                if k not in self.opt_cols:
                    design[k] = v

            # Always compute pre-snap (continuous) physical values for CSV export
            pre_snap = {}
            for col, lv in zip(self.opt_cols, final_log_vals):
                if col in self.log10_cols:
                    pre_snap[col] = float(10 ** lv)
                elif col in self.log2_cols:
                    pre_snap[col] = float(2 ** lv)
                else:
                    pre_snap[col] = float(lv)

            # Second forward pass using snapped values — gives the honest post-snap surrogate prediction
            snapped_log_vals = torch.zeros(len(self.opt_cols), device=self.device)
            for j, col in enumerate(self.opt_cols):
                val = float(design.get(col, 0.0))
                if col in self.log10_cols:
                    snapped_log_vals[j] = float(np.log10(max(val, 1e-12)))
                elif col in self.log2_cols:
                    snapped_log_vals[j] = float(np.log2(max(val, 1)))
                else:
                    snapped_log_vals[j] = val

            cap_log10_snap = snapped_log_vals[self.opt_cols.index("capacity_kb")]
            cap_phys_snap  = 10 ** cap_log10_snap

            x_snap = base_x.clone()
            for col, lv in zip(self.opt_cols, snapped_log_vals):
                if col in dyn_idx:
                    x_snap[dyn_idx[col]] = lv

            if "derived_sqrt_capacity" in dyn_idx:
                x_snap[dyn_idx["derived_sqrt_capacity"]] = torch.sqrt(cap_phys_snap)

            if "data_stacked_die_count" in self.opt_cols:
                stk_phys_snap = 2 ** snapped_log_vals[self.opt_cols.index("data_stacked_die_count")]
            else:
                stk_phys_snap = torch.tensor(float(fixed_context.get("data_stacked_die_count", 1.0)), device=self.device)

            if "word_width_bits" in self.opt_cols:
                ww_phys_snap = 2 ** snapped_log_vals[self.opt_cols.index("word_width_bits")]
            else:
                ww_phys_snap = torch.tensor(2 ** float(fixed_context.get("word_width_bits", 6.0)), device=self.device)

            if "derived_cap_per_die" in dyn_idx:
                x_snap[dyn_idx["derived_cap_per_die"]]  = cap_phys_snap / stk_phys_snap
            if "derived_rows_per_die" in dyn_idx:
                x_snap[dyn_idx["derived_rows_per_die"]] = (cap_phys_snap * 1024) / (ww_phys_snap * stk_phys_snap)

            if self.tech == "SRAM" and "CellInput_SRAMCellNMOSWidth (F)" in self.opt_cols:
                _update_sram_cell_features(x_snap, dyn_idx, snapped_log_vals, self.opt_cols, fixed_context, self.device)

            for rm in ["HP", "LOP", "LSTP"]:
                col = f"device_roadmap_{rm}_x_log10_cap"
                if col in dyn_idx:
                    x_snap[dyn_idx[col]] = float(fixed_context.get(f"device_roadmap_{rm}", 0.0)) * cap_log10_snap

            x_snap_scaled = ((x_snap - self.means) / self.stds).unsqueeze(0)
            pred_snap, _ = self.model.forward_with_feasibility(x_snap_scaled)
            snapped_ppa = 10 ** pred_snap.cpu().numpy()[0]
            active_short = [TARGET_SHORT_LABELS[TARGET_KEYS.index(k)] for k in self._active_keys]
            snapped_ppa_dict = {label: snapped_ppa[i] for i, label in enumerate(active_short)}

            if verbose:
                col_w = max(len(c) for c in self.opt_cols) + 2
                print(f"\n  {'Parameter':<{col_w}}  {'Pre-snap (continuous)':>22}  {'Post-snap (physical)':>20}")
                print(f"  {'-'*col_w}  {'':->22}  {'':->20}")
                for col in self.opt_cols:
                    pre  = pre_snap.get(col, float("nan"))
                    post = design.get(col, float("nan"))
                    changed = "  *" if abs(pre - post) > 1e-6 * (abs(pre) + 1e-12) else ""
                    print(f"  {col:<{col_w}}  {pre:>22.6g}  {post:>20.6g}{changed}")
                print()

            pred_ppa = 10 ** pred.detach().cpu().numpy()[0]
            ppa_dict = {label: pred_ppa[i] for i, label in enumerate(active_short)}

        return design, ppa_dict, pre_snap, snapped_ppa_dict


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DESTINY Inverse Design Optimizer")
    p.add_argument("--tech",           default="SRAM")
    p.add_argument("--target-read-latency",    type=float, help="Target read latency (ns)")
    p.add_argument("--target-write-latency",   type=float, help="Target write latency (ns)")
    p.add_argument("--target-refresh-latency", type=float, help="Target refresh latency (ns)")
    p.add_argument("--target-area",            type=float, help="Target cache area (mm2)")
    p.add_argument("--target-hit-energy",      type=float, help="Target hit energy (nJ)")
    p.add_argument("--target-write-energy",    type=float, help="Target write energy (nJ)")
    p.add_argument("--target-refresh-energy",  type=float, help="Target refresh energy (nJ)")
    p.add_argument("--target-leakage",         type=float, help="Target leakage power (mW)")
    p.add_argument("--node",           type=int, default=32, choices=[22, 32, 45, 65])
    p.add_argument("--roadmap",        default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--temperature",    type=float, default=350.0)
    p.add_argument("--output",         default=None, help="CSV to append results to")
    p.add_argument("--verbose-opt",    action="store_true", help="Print pre/post-snap parameter table")
    args = p.parse_args()

    targets = {k: v for k, v in [
        ("cache_area_mm2",           args.target_area),
        ("cache_hit_latency_ns",     args.target_read_latency if args.target_read_latency is not None else args.target_latency),
        ("cache_write_latency_ns",    args.target_write_latency),
        ("cache_refresh_latency_ns",  args.target_refresh_latency),
        ("cache_hit_energy_nJ",      args.target_hit_energy),
        ("cache_write_energy_nJ",    args.target_write_energy if args.target_write_energy is not None else args.target_energy),
        ("cache_refresh_energy_nJ",  args.target_refresh_energy),
        ("cache_leakage_mW",         args.target_leakage),
    ] if v is not None}

    if not targets:
        p.error("Specify at least one target via --target-read-latency / --target-area / etc.")

    context = {
        f"process_node_nm_{args.node}":   1.0,
        f"device_roadmap_{args.roadmap}": 1.0,
        "temperature_K":                  args.temperature,
    }

    design, ppa, pre_snap, snapped_ppa = InverseOptimizer(args.tech).optimize(
        targets, context, verbose=args.verbose_opt)

    row = {"tech": args.tech, "node_nm": args.node,
           "roadmap": args.roadmap, "temperature_K": args.temperature}
    row.update(design)
    row.update({
        f"pred_{k}": ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)
    })
    row.update({
        f"post_snap_pred_{k}": snapped_ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)
    })
    row.update({
        f"target_{k}": targets.get(k, float("nan")) for k in TARGET_KEYS
    })
    row.update({f"pre_snap_{k}": v for k, v in pre_snap.items()})

    df_out = pd.DataFrame([row])
    if args.output:
        df_out.to_csv(args.output, mode="a", index=False, header=not os.path.exists(args.output))
        print(f"Result appended to {args.output}", file=sys.stderr)
    else:
        print(df_out.to_csv(index=False), end="")
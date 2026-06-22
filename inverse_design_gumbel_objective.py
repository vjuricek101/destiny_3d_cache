#!/usr/bin/env python3
"""
inverse_design_gumbel_objective.py — ST Gumbel-Softmax inverse design optimizer directly minimizing objectives.

Each architectural parameter is a categorical variable over an explicit discrete
vocabulary.  During the inner loop:
  1. Sample Gumbel noise:  g = -log(-log(U + ε) + ε),  U ~ Uniform(0,1)
  2. Compute soft Gumbel-Softmax relaxation:  y = softmax((logits + g) / τ)
  3. Apply the Straight-Through trick (Jang et al., ICLR 2017):
       y_hard = one_hot(argmax(y))
       y_st   = (y_hard - y).detach() + y
  4. Scalar value fed to the surrogate = inner product of y_st and the vocab tensor.

Temperature τ is annealed exponentially from GUMBEL_TAU_START → GUMBEL_TAU_END
so the distribution hardens progressively over optimization steps.
"""

import math
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import pickle

import torch
import numpy as np
import pandas as pd
from train_model import PPA_MLP, CATEGORICAL_COLS, LOG_NUMERIC_COLS, LOG2_CFG_COLS
import sys
from destiny_utils import (
    TARGET_COLS as TARGET_KEYS,
    TARGET_SHORT_LABELS,
    get_active_targets,
    validate_cache_geometry,
    build_fixed_context,
    parse_fixed_arg,
    build_gumbel_vocabs,
    build_dyn_idx,
    BASE_ARCH_COLS,
    BASE_ARCH_BOUNDS,
    DATA_PARAM_BOUNDS_LOG2,
    DATA_PARAM_BOUNDS_LINEAR,
    SRAM_CELL_BOUNDS_LOG10,
    SRAM_CELL_BOUNDS_LINEAR,
)
from inverse_physics_validity import (
    compute_physics_penalties,
    check_post_snap_partition
)

# ── Gumbel hyper-parameters ───────────────────────────────────────────────────
GUMBEL_TAU_START = 5.0   # hot  → broad distribution, low gradient variance
GUMBEL_TAU_END   = 0.5   # cold → near-discrete, higher variance


# ── Core Gumbel-Softmax primitive ─────────────────────────────────────────────

def gumbel_softmax_st(logits: torch.Tensor, tau: float):
    """Straight-Through Gumbel-Softmax sample (Jang et al., ICLR 2017, Eqs. 1–4).

    Forward : hard one-hot via argmax.
    Backward: gradients flow through the soft Gumbel-Softmax relaxation y.

    Returns
    -------
    y_st : shape [K] — STE one-hot with gradient w.r.t. logits
    y    : shape [K] — soft sample (for gradient computation only)
    """
    eps = 1e-20
    u   = torch.rand_like(logits)
    g   = -torch.log(-torch.log(u + eps) + eps)            # Gumbel(0,1) noise
    y   = torch.softmax((logits + g) / tau, dim=-1)         # soft relaxation
    y_hard = torch.zeros_like(logits).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
    y_st   = (y_hard - y).detach() + y                      # STE
    return y_st, y


# ── Temperature schedule ──────────────────────────────────────────────────────

def _anneal_tau(step: int, total_steps: int) -> float:
    """Exponential annealing from GUMBEL_TAU_START → GUMBEL_TAU_END."""
    if total_steps <= 1:
        return GUMBEL_TAU_END
    frac    = step / (total_steps - 1)
    log_tau = (1.0 - frac) * math.log(GUMBEL_TAU_START) + frac * math.log(GUMBEL_TAU_END)
    return float(math.exp(log_tau))


def _update_sram_cell_features(x, dyn_idx, encoded_vals_dict, fixed_context, device):
    """Compute SRAM cell derived features and write them into feature vector x (in-place).

    encoded_vals_dict holds differentiable scalar tensors (inner products of y_st and
    their vocab tensors), so autograd graphs remain intact back to logits.
    """
    wn  = 10 ** encoded_vals_dict["CellInput_SRAMCellNMOSWidth (F)"]
    wp  = 10 ** encoded_vals_dict["CellInput_SRAMCellPMOSWidth (F)"]
    wac = 10 ** encoded_vals_dict["CellInput_AccessCMOSWidth (F)"]

    rv_key = "CellInput_ReadVoltage (V)"
    rv = encoded_vals_dict.get(rv_key,
         torch.tensor(float(fixed_context.get(rv_key, 1.0)), device=device))

    # Mirrors the formula and clamping in derive_sram_physical_params.
    cell_area = torch.clamp(55.0 + 30.0 * torch.maximum(wn, wac) + 20.0 * (wp + 0.5), 40.0, 200.0)

    if "CellInput_CellArea (F^2)" in dyn_idx:
        x[dyn_idx["CellInput_CellArea (F^2)"]] = torch.log10(cell_area)
    if "derived_sqrt_area" in dyn_idx:
        x[dyn_idx["derived_sqrt_area"]] = torch.sqrt(cell_area)
    if "derived_read_v_sq" in dyn_idx:
        x[dyn_idx["derived_read_v_sq"]] = rv ** 2

    if "CellInput_MinSenseVoltage (mV)" in dyn_idx:
        a_vth = 3.0
        for k in fixed_context:
            if k.startswith("process_node_nm_") and float(fixed_context[k]) > 0.5:
                node  = int(k.split("_")[-1])
                a_vth = {65: 5.0, 45: 4.0, 32: 3.0, 22: 2.5}.get(node, 3.0)
                break
        v_sense = torch.clamp(6.0 * a_vth / torch.sqrt(2.0 * wac), 5.0, 80.0)
        x[dyn_idx["CellInput_MinSenseVoltage (mV)"]] = v_sense


def _snap_design_gumbel(vocabs: dict, logits_dict: dict) -> dict:
    """Convert final logits to physical design values via argmax over each vocabulary."""
    design = {}
    for col, logit in logits_dict.items():
        best_idx = int(logit.detach().argmax().item())
        enc_val  = float(vocabs[col][best_idx].item())
        if col == "capacity_kb":
            design[col] = int(round(10 ** enc_val))
        elif col in ("word_width_bits", "associativity", "data_stacked_die_count"):
            design[col] = int(round(2 ** enc_val))
        elif "mux_" in col:
            design[col] = int(round(2 ** enc_val))
        elif "_mat_" in col or "num_row_mat" in col or "num_col_mat" in col:
            design[col] = int(round(2 ** enc_val))
        elif "_subarray_" in col:
            design[col] = int(round(enc_val))
        elif col in SRAM_CELL_BOUNDS_LOG10:
            design[col] = float(round(10 ** enc_val, 4))
        elif col in SRAM_CELL_BOUNDS_LINEAR:
            design[col] = float(round(enc_val, 4))
        else:
            design[col] = enc_val
    return design


# ── Optimizer ─────────────────────────────────────────────────────────────────

class InverseOptimizerGumbel:
    """Inverse design optimizer using Straight-Through Gumbel-Softmax to directly minimize PPA objectives.
    """

    def __init__(self, tech: str):
        self.tech   = tech
        self.device = torch.device("cpu")

        model_dir = f"model_output/{tech.lower()}_feasibility"

        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        sd        = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        hidden    = sd["input_proj.weight"].shape[0]
        n_blocks  = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
        n_targets = sd["output_head.weight"].shape[0]
        self.model = PPA_MLP(len(self.feature_cols), hidden_dim=hidden,
                             n_blocks=n_blocks, n_targets=n_targets).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        # Active target keys — ordered subset used during training for this tech.
        self._active_keys = get_active_targets(self.tech)

        self.means = torch.tensor(self.scaler.mean_,  dtype=torch.float32, device=self.device)
        self.stds  = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

        # Candidate opt columns — pruned per-call in optimize() via fixed_context.
        self._all_opt_cols    = []
        self._all_log10_cols  = []
        self._all_log2_cols   = []
        self._all_linear_cols = []

        for col, _ in zip(BASE_ARCH_COLS, BASE_ARCH_BOUNDS):
            if col in self.feature_cols:
                self._all_opt_cols.append(col)
                (self._all_log10_cols if col == "capacity_kb" else self._all_log2_cols).append(col)

        for col in DATA_PARAM_BOUNDS_LOG2:
            if col in self.feature_cols:
                self._all_opt_cols.append(col)
                self._all_log2_cols.append(col)

        for col in DATA_PARAM_BOUNDS_LINEAR:
            if col in self.feature_cols:
                self._all_opt_cols.append(col)
                self._all_linear_cols.append(col)

        if self.tech == "SRAM":
            for col in SRAM_CELL_BOUNDS_LOG10:
                if col in self.feature_cols:
                    self._all_opt_cols.append(col)
                    self._all_log10_cols.append(col)
            for col in SRAM_CELL_BOUNDS_LINEAR:
                if col in self.feature_cols:
                    self._all_opt_cols.append(col)
                    self._all_linear_cols.append(col)

        self.categorical_vocabs = {}
        for cat in CATEGORICAL_COLS:
            if cat in ["mem_cell_type", "device_roadmap", "process_node_nm"]:
                continue
            vals = []
            prefix = cat + "_"
            for feat in self.feature_cols:
                if feat.startswith(prefix):
                    vals.append(feat[len(prefix):])
            if len(vals) > 1:
                self.categorical_vocabs[cat] = vals

    def _objective_mask(self, objectives: list):
        """Return a binary weight mask aligned to the active keys of the model output head."""
        if not objectives or "all" in objectives or objectives == ["all"]:
            resolved = set(self._active_keys)
        else:
            resolved = set()
            for obj in objectives:
                if obj in self._active_keys:
                    resolved.add(obj)
                elif obj in TARGET_SHORT_LABELS:
                    idx = TARGET_SHORT_LABELS.index(obj)
                    key = TARGET_KEYS[idx]
                    if key in self._active_keys:
                        resolved.add(key)
        
        weights = [1.0 if k in resolved else 0.0 for k in self._active_keys]
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _build_feature_vector(
        self,
        base_x: torch.Tensor,
        dyn_idx: dict,
        encoded_vals_dict: dict,
        opt_cols: list,
        fixed_context: dict,
        cap_enc: torch.Tensor,
        cat_encoded_dict: dict = None,
    ) -> torch.Tensor:
        """Construct the full feature vector for the surrogate model."""
        x = base_x.clone()

        for col, enc_val in encoded_vals_dict.items():
            if col in dyn_idx:
                x[dyn_idx[col]] = enc_val

        if cat_encoded_dict:
            for cat, y_st in cat_encoded_dict.items():
                vals = self.categorical_vocabs[cat]
                for i, val in enumerate(vals):
                    col_name = f"{cat}_{val}"
                    if col_name in dyn_idx:
                        x[dyn_idx[col_name]] = y_st[i]

        cap_phys = 10.0 ** cap_enc
        if "derived_sqrt_capacity" in dyn_idx:
            x[dyn_idx["derived_sqrt_capacity"]] = torch.sqrt(cap_phys)

        die_key = "data_stacked_die_count"
        stk_phys = (2.0 ** encoded_vals_dict[die_key] if die_key in encoded_vals_dict
                    else torch.tensor(float(fixed_context.get(die_key, 1.0)), device=self.device))

        ww_key = "word_width_bits"
        ww_phys = (2.0 ** encoded_vals_dict[ww_key] if ww_key in encoded_vals_dict
                   else torch.tensor(float(fixed_context.get(ww_key, 64.0)), device=self.device))

        if "derived_cap_per_die" in dyn_idx:
            x[dyn_idx["derived_cap_per_die"]]  = cap_phys / stk_phys
        if "derived_rows_per_die" in dyn_idx:
            x[dyn_idx["derived_rows_per_die"]] = (cap_phys * 1024.0) / (ww_phys * stk_phys)

        sram_width_cols = set(SRAM_CELL_BOUNDS_LOG10) | set(SRAM_CELL_BOUNDS_LINEAR)
        if self.tech == "SRAM" and sram_width_cols & set(encoded_vals_dict):
            _update_sram_cell_features(x, dyn_idx, encoded_vals_dict, fixed_context, self.device)

        for rm in ["HP", "LOP", "LSTP"]:
            col = f"device_roadmap_{rm}_x_log10_cap"
            if col in dyn_idx:
                x[dyn_idx[col]] = float(fixed_context.get(f"device_roadmap_{rm}", 0.0)) * cap_enc

        return x

    def optimize(
        self,
        objectives: list,
        fixed_context: dict,
        steps: int = 300,
        n_restarts: int = 4,
        objective_weights=None,
        verbose: bool = False,
    ):
        """ST Gumbel-Softmax gradient-based inverse design directly minimizing objectives.

        Parameters
        ----------
        objectives        : list of metrics (column names or short labels) to minimize
        fixed_context     : {col: value} — columns pinned and not optimised
        steps             : gradient steps per restart
        n_restarts        : number of random restarts
        objective_weights : optional override for per-metric loss weights
        verbose           : if True, print per-parameter table at the end

        Returns
        -------
        design           : {col: physical_value} — snapped hardware design
        ppa_dict         : {label: predicted_value} — continuous-param PPA
        pre_snap         : {col: pre-snap physical value}
        snapped_ppa_dict : {label: predicted_value} — post-snap PPA
        """
        # Drop columns pinned by fixed_context.
        fixed_keys       = set(fixed_context.keys())
        self.opt_cols    = [c for c in self._all_opt_cols    if c not in fixed_keys]
        self.log10_cols  = [c for c in self._all_log10_cols  if c not in fixed_keys]
        self.log2_cols   = [c for c in self._all_log2_cols   if c not in fixed_keys]
        self.linear_cols = [c for c in self._all_linear_cols if c not in fixed_keys]
        self.opt_cats    = [c for c in self.categorical_vocabs if c not in fixed_keys]

        vocabs = {k: v.to(self.device) for k, v in build_gumbel_vocabs(self.opt_cols, self.tech).items()}

        if "word_width_bits" in vocabs:
            cap_kb = float(fixed_context.get("capacity_kb", 64.0))
            assoc = float(fixed_context.get("associativity", 4.0))
            max_ww_bits = (cap_kb * 8192.0) / assoc
            phys_ww = 2.0 ** vocabs["word_width_bits"]
            valid_mask = phys_ww <= max_ww_bits
            if not valid_mask.any():
                valid_mask[0] = True
            vocabs["word_width_bits"] = vocabs["word_width_bits"][valid_mask]

        w_tensor = self._objective_mask(objectives)
        if objective_weights is not None:
            if isinstance(objective_weights, dict):
                w_tensor = torch.tensor(
                    [objective_weights.get(k, 1.0 if k in objectives else 0.0) for k in TARGET_KEYS],
                    dtype=torch.float32, device=self.device)
            else:
                w_tensor = torch.tensor(objective_weights, dtype=torch.float32, device=self.device)

        # If capacity_kb is fixed, pre-compute its encoded value so _build_feature_vector works.
        _cap_fixed = "capacity_kb" in fixed_keys
        if _cap_fixed:
            _cap_kb_val = float(fixed_context["capacity_kb"])
            _cap_enc_fixed = torch.tensor(math.log10(_cap_kb_val), dtype=torch.float32, device=self.device)

        dyn_idx = build_dyn_idx(self.feature_cols, self.opt_cols, self.opt_cats, self.categorical_vocabs)

        # Static base feature vector; dynamic cols filled each step.
        base_x = torch.zeros(len(self.feature_cols), device=self.device)
        for i, c in enumerate(self.feature_cols):
            if c not in dyn_idx:
                raw_val = float(fixed_context.get(c, 0.0))
                if c in LOG_NUMERIC_COLS:
                    base_x[i] = math.log10(max(1e-12, raw_val))
                elif c in LOG2_CFG_COLS:
                    base_x[i] = math.log2(max(1.0, raw_val))
                else:
                    base_x[i] = raw_val

        best_loss, best_logits, best_cat_logits, best_pred = float("inf"), None, None, None

        for restart in range(max(1, n_restarts)):
            # Restart 0: uniform prior (zero logits); subsequent: small random init.
            logits_dict = {
                col: (torch.zeros(vocabs[col].shape[0], device=self.device)
                      if restart == 0
                      else torch.randn(vocabs[col].shape[0], device=self.device) * 0.5
                      ).requires_grad_(True)
                for col in self.opt_cols
            }
            cat_logits_dict = {
                cat: (torch.zeros(len(self.categorical_vocabs[cat]), device=self.device)
                      if restart == 0
                      else torch.randn(len(self.categorical_vocabs[cat]), device=self.device) * 0.5
                      ).requires_grad_(True)
                for cat in self.opt_cats
            }
            
            inner_opt = torch.optim.Adam(list(logits_dict.values()) + list(cat_logits_dict.values()), lr=0.05)

            pred = None
            for step in range(steps):
                inner_opt.zero_grad()
                tau = _anneal_tau(step, steps)

                # Sample Gumbel-Softmax; accumulate differentiable scalar per column.
                encoded_vals_dict = {
                    col: (gumbel_softmax_st(logits_dict[col], tau)[0] * vocabs[col]).sum()
                    for col in self.opt_cols
                }
                
                cat_encoded_dict = {
                    cat: gumbel_softmax_st(cat_logits_dict[cat], tau)[0]
                    for cat in self.opt_cats
                }

                cap_enc = _cap_enc_fixed if _cap_fixed else encoded_vals_dict["capacity_kb"]
                x       = self._build_feature_vector(
                    base_x, dyn_idx, encoded_vals_dict, self.opt_cols, fixed_context, cap_enc, cat_encoded_dict
                )

                x_scaled = ((x - self.means) / self.stds).unsqueeze(0)
                pred, p_feas    = self.model.forward_with_feasibility(x_scaled)
                learned_penalty = 50.0 * (1.0 - p_feas.squeeze())

                physics_penalty = compute_physics_penalties(encoded_vals_dict, fixed_context, cap_enc, self.device)

                # Direct minimization of the selected predictions (clamped and scaled to match penalty gradients scale)
                pred_bounded = torch.clamp(pred, min=-6.0)
                loss = 100.0 * (w_tensor * pred_bounded).sum() + learned_penalty + physics_penalty
                loss.backward()
                inner_opt.step()

            with torch.no_grad():
                final_loss = loss.item()
            if final_loss < best_loss:
                best_loss   = final_loss
                best_logits = {k: v.detach().clone() for k, v in logits_dict.items()}
                best_cat_logits = {k: v.detach().clone() for k, v in cat_logits_dict.items()}
                best_pred   = pred.detach()

        # ── Post-optimisation: deterministic snap (argmax, no Gumbel noise) ──
        with torch.no_grad():
            design = _snap_design_gumbel(vocabs, best_logits)

            for cat in self.opt_cats:
                best_idx = int(best_cat_logits[cat].argmax().item())
                design[cat] = self.categorical_vocabs[cat][best_idx]

            # Associativity was zero-variance in training; always default to 4.
            if "associativity" not in design:
                design["associativity"] = 4

            # Enforce SRAM transistor ratio constraints: γ (wp < wac), β (wn ≥ 2·wac).
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

            # Merge fixed_context values not already in design.
            for k, v in fixed_context.items():
                if k not in self.opt_cols:
                    design[k] = v

            check_post_snap_partition(design, fixed_context)

            # Pre-snap: physical values decoded from argmax vocab entries.
            pre_snap = {}
            for col in self.opt_cols:
                best_idx = int(best_logits[col].argmax().item())
                enc_val  = float(vocabs[col][best_idx].item())
                if col in self.log10_cols:
                    pre_snap[col] = float(10 ** enc_val)
                elif col in self.log2_cols:
                    pre_snap[col] = float(2 ** enc_val)
                else:
                    pre_snap[col] = enc_val

            for cat in self.opt_cats:
                best_idx = int(best_cat_logits[cat].argmax().item())
                pre_snap[cat] = self.categorical_vocabs[cat][best_idx]

            # Second forward pass on snapped values — honest post-snap surrogate prediction.
            snapped_encoded = {
                col: torch.tensor(float(vocabs[col][int(best_logits[col].argmax().item())].item()),
                                  device=self.device)
                for col in self.opt_cols
            }
            snapped_cat_encoded = {
                cat: torch.zeros(len(self.categorical_vocabs[cat]), device=self.device).scatter_(0, torch.tensor(int(best_cat_logits[cat].argmax().item()), device=self.device), 1.0)
                for cat in self.opt_cats
            }
            
            cap_enc_snap = _cap_enc_fixed if _cap_fixed else snapped_encoded["capacity_kb"]
            x_snap       = self._build_feature_vector(
                base_x, dyn_idx, snapped_encoded, self.opt_cols, fixed_context, cap_enc_snap, snapped_cat_encoded
            )
            x_snap_scaled = ((x_snap - self.means) / self.stds).unsqueeze(0)
            pred_snap, _  = self.model.forward_with_feasibility(x_snap_scaled)

            active_short     = [TARGET_SHORT_LABELS[TARGET_KEYS.index(k)] for k in self._active_keys]
            pred_snap_clipped = np.clip(pred_snap.cpu().numpy()[0], -35.0, 35.0)
            snapped_ppa      = 10 ** pred_snap_clipped
            snapped_ppa_dict = {label: snapped_ppa[i] for i, label in enumerate(active_short)}

            best_pred_clipped = np.clip(best_pred.cpu().numpy()[0], -35.0, 35.0)
            pred_ppa = 10 ** best_pred_clipped
            ppa_dict = {label: pred_ppa[i] for i, label in enumerate(active_short)}

            if verbose:
                col_w = max([len(c) for c in self.opt_cols + self.opt_cats] + [0]) + 2
                print(f"\n  {'Parameter':<{col_w}}  {'Gumbel argmax (physical)':>25}  {'Post-snap (physical)':>20}")
                print(f"  {'-'*col_w}  {'':->25}  {'':->20}")
                for col in self.opt_cols + self.opt_cats:
                    pre  = pre_snap.get(col, float("nan"))
                    post = design.get(col, float("nan"))
                    if isinstance(pre, float):
                        print(f"  {col:<{col_w}}  {pre:>25.6g}  {post:>20.6g}")
                    else:
                        print(f"  {col:<{col_w}}  {str(pre):>25}  {str(post):>20}")
                print()

        return design, ppa_dict, pre_snap, snapped_ppa_dict


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DESTINY Inverse Design Optimizer — ST Gumbel-Softmax Direct Objective Minimization")
    p.add_argument("--tech",          default="SRAM")
    p.add_argument("--targets",       nargs="+", default=["all"], help="PPA objectives/targets to minimize (default: all)")
    p.add_argument("--capacity-kb",   type=float, default=64.0, help="Fixed capacity (KB)")
    p.add_argument("--associativity", type=int, default=4, help="Fixed associativity")
    p.add_argument("--word-width-bits", type=int, default=None, help="Fixed word width (bits)")
    p.add_argument("--node",        type=int, default=32, choices=[22, 32, 45, 65])
    p.add_argument("--roadmap",     default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--temperature", type=float, default=350.0)
    p.add_argument("--steps",       type=int,   default=300)
    p.add_argument("--restarts",    type=int,   default=4)
    p.add_argument(
        "--fix", nargs="*", default=[], metavar="KEY=VALUE", type=parse_fixed_arg,
        help="Pin design parameters as optimizer constants. "
             "Values are auto-coerced to int, float, or str. "
             "E.g. --fix capacity_kb=64 associativity=8"
    )
    p.add_argument("--output",      default=None, help="CSV to append results to")
    p.add_argument("--verbose-opt", action="store_true",
                   help="Print Gumbel argmax / post-snap parameter table")
    args = p.parse_args()

    validate_cache_geometry(args.capacity_kb, args.associativity, args.word_width_bits)

    fixed   = dict(args.fix)
    # Add input flags directly to fixed parameter constraints
    fixed["capacity_kb"] = args.capacity_kb
    fixed["associativity"] = args.associativity
    if args.word_width_bits is not None:
        fixed["word_width_bits"] = args.word_width_bits

    context = build_fixed_context(args.node, args.roadmap, args.temperature, **fixed)

    optimizer = InverseOptimizerGumbel(args.tech)
    design, ppa, pre_snap, snapped_ppa = optimizer.optimize(
        args.targets, context,
        steps=args.steps, n_restarts=args.restarts,
        verbose=args.verbose_opt,
    )

    row = {"tech": args.tech, "node_nm": args.node,
           "roadmap": args.roadmap, "temperature_K": args.temperature}
    row.update(design)
    row.update({f"pred_{k}": ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)})
    row.update({f"post_snap_pred_{k}": snapped_ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)})
    row.update({f"target_{k}": float("nan") for k in TARGET_KEYS})
    row.update({f"pre_snap_{k}": v for k, v in pre_snap.items()})
    row["objectives"] = ",".join(args.targets)

    df_out = pd.DataFrame([row])
    if args.output:
        df_out.to_csv(args.output, mode="a", index=False,
                      header=not os.path.exists(args.output))
        print(f"Result appended to {args.output}", file=sys.stderr)
    else:
        print(df_out.to_csv(index=False), end="")

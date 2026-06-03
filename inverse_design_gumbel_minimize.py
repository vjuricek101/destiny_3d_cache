#!/usr/bin/env python3
"""
inverse_design_gumbel_minimize.py — ST Gumbel-Softmax inverse design optimizer (minimize variant).

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
import argparse
import json
import pickle

import torch
import numpy as np
import pandas as pd
from train_model import PPA_MLP, CATEGORICAL_COLS
import sys
from typing import Optional
from destiny_utils import (
    TARGET_COLS as TARGET_KEYS,
    TARGET_SHORT_LABELS,
    get_active_targets,
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

class InverseOptimizerGumbelMinimize:
    """Inverse design optimizer using Straight-Through Gumbel-Softmax (minimize variant).

    Instead of matching target values, this optimizer directly minimises the sum of
    log10-predicted values for a requested set of objectives.  The caller supplies
    a list of metric names; no target magnitudes are needed.

    optimize() returns (design, ppa_dict, pre_snap, snapped_ppa_dict).
    """

    def __init__(self, tech: str, model_dir: Optional[str] = None):
        self.tech   = tech
        self.device = torch.device("cpu")

        if model_dir is None:
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
        """Construct the full feature vector for the surrogate model.

        Parameters
        ----------
        base_x            : static feature vector (fixed_context values)
        dyn_idx           : {col_name: feature_index}
        encoded_vals_dict : {col: differentiable scalar tensor in encoded space}
        opt_cols          : ordered list of optimised column names
        fixed_context     : raw context dict
        cap_enc           : log10(capacity_kb) scalar tensor

        Returns
        -------
        x : 1-D feature tensor, differentiable w.r.t. logits
        """
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
                   else torch.tensor(2.0 ** float(fixed_context.get(ww_key, 6.0)), device=self.device))

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
        objectives: list,        # e.g. ["cache_hit_latency_ns", "cache_hit_energy_nJ"]
        fixed_context: dict,
        steps: int = 300,
        n_restarts: int = 4,
        objective_weights=None,  # optional list/dict of per-objective weights for Pareto attenuation
        verbose: bool = False,
    ):
        """ST Gumbel-Softmax gradient-based inverse design with multi-start (minimize variant).

        Parameters
        ----------
        objectives        : list of metric keys to minimise (must be active for this tech)
        fixed_context     : {col: value} — columns pinned and not optimised.
                            Pass ``"capacity_kb": <int>`` here to fix capacity instead of
                            optimizing it (strongly recommended to prevent collapse to 2 KB).
        steps             : gradient steps per restart
        n_restarts        : number of random restarts
        objective_weights : optional per-objective weights that control which region of the
                            Pareto frontier is favoured.  Can be:
                              - a list of floats, one per entry in *objectives*
                              - a dict {metric_key: float}
                            Larger weights push the optimizer harder toward minimizing that
                            metric.  Defaults to equal weight (1.0) for all objectives.
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

        # Map each requested metric to its index in the model output head.
        # Raises KeyError immediately if a metric is not active for this tech,
        # rather than silently producing wrong gradients.
        obj_indices = []
        for k in objectives:
            if k not in self._active_keys:
                raise KeyError(
                    f"Metric '{k}' is not active for tech={self.tech}. "
                    f"Active metrics: {self._active_keys}"
                )
            obj_indices.append(self._active_keys.index(k))
        obj_indices_t = torch.tensor(obj_indices, dtype=torch.long, device=self.device)

        # Per-objective weights for Pareto attenuation.
        if objective_weights is None:
            obj_w = torch.ones(len(objectives), dtype=torch.float32, device=self.device)
        elif isinstance(objective_weights, dict):
            obj_w = torch.tensor(
                [objective_weights.get(k, 1.0) for k in objectives],
                dtype=torch.float32, device=self.device,
            )
        else:
            obj_w = torch.tensor(objective_weights, dtype=torch.float32, device=self.device)
        # Normalize so weights sum to len(objectives) — preserves approximate loss scale.
        obj_w = obj_w * (len(objectives) / obj_w.sum().clamp(min=1e-8))

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
                base_x[i] = float(fixed_context.get(c, 0.0))

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

                # Use fixed capacity encoding if capacity_kb is pinned in fixed_context.
                cap_enc = _cap_enc_fixed if _cap_fixed else encoded_vals_dict["capacity_kb"]
                x       = self._build_feature_vector(
                    base_x, dyn_idx, encoded_vals_dict, self.opt_cols, fixed_context, cap_enc, cat_encoded_dict
                )

                x_scaled = ((x - self.means) / self.stds).unsqueeze(0)
                pred, p_feas    = self.model.forward_with_feasibility(x_scaled)
                learned_penalty = 50.0 * (1.0 - p_feas.squeeze())

                # ── DESTINY partition constraint penalty ──────────────────────────────────────
                # Enforces: word_width >= mat_r * mat_c * sub_r * sub_c
                # From DESTINY main.cpp: blockSize / (active_partition_product) == 0
                # causes every candidate config to be silently skipped (numSolutions = 0).
                #
                # Encoding conventions in encoded_vals_dict:
                #   word_width_bits, data_num_active_mat_per_row/col  → log2-encoded
                #   data_num_active_subarray_per_row/col              → linear (vocab = {1.0, 2.0})
                #
                # We convert everything to physical linear space for the penalty so the
                # gradient is directly interpretable and avoids log2 singularities.

                _ww_enc = encoded_vals_dict.get(
                    "word_width_bits",
                    torch.tensor(math.log2(float(fixed_context.get("word_width_bits", 64))),
                                device=self.device)
                )
                _mat_r_enc = encoded_vals_dict.get(
                    "data_num_active_mat_per_row",
                    torch.tensor(0.0, device=self.device)   # log2(1) = 0 → physical default = 1
                )
                _mat_c_enc = encoded_vals_dict.get(
                    "data_num_active_mat_per_col",
                    torch.tensor(0.0, device=self.device)   # log2(1) = 0 → physical default = 1
                )
                # Subarray terms: linear vocab {1.0, 2.0}, already physical
                _sub_r_phys = encoded_vals_dict.get(
                    "data_num_active_subarray_per_row",
                    torch.tensor(1.0, device=self.device)
                )
                _sub_c_phys = encoded_vals_dict.get(
                    "data_num_active_subarray_per_col",
                    torch.tensor(1.0, device=self.device)
                )

                _ww_phys        = 2.0 ** _ww_enc
                _mat_r_phys     = 2.0 ** _mat_r_enc
                _mat_c_phys     = 2.0 ** _mat_c_enc
                _partition_phys = _mat_r_phys * _mat_c_phys * _sub_r_phys * _sub_c_phys
                
                partition_penalty = 50.0 * torch.relu(_partition_phys - _ww_phys) / _ww_phys

                # pred shape: [1, n_active_targets], values are log10(physical).
                # Weighted sum of log10 predictions for the requested objectives.
                # Weights > 1 push harder toward minimizing that metric (Pareto attenuation).
                task_loss = (obj_w * pred[0, obj_indices_t]).sum()
                loss = task_loss + learned_penalty + partition_penalty
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

            # ── Post-snap partition constraint check ──────────────────────────────────────
            # Mirrors DESTINY main.cpp: blockSize / (active_mat_row * active_mat_col *
            # active_sub_row * active_sub_col) == 0 causes silent discard of all configs.
            # For normal/sequential cache access mode, blockSize == word_width_bits.
            _ps_ww    = int(design.get("word_width_bits",
                                        fixed_context.get("word_width_bits", 64)))
            _ps_mat_r = int(design.get("data_num_active_mat_per_row", 1))
            _ps_mat_c = int(design.get("data_num_active_mat_per_col", 1))
            _ps_sub_r = int(design.get("data_num_active_subarray_per_row", 1))
            _ps_sub_c = int(design.get("data_num_active_subarray_per_col", 1))
            _ps_partition = _ps_mat_r * _ps_mat_c * _ps_sub_r * _ps_sub_c

            if _ps_partition > _ps_ww:
                print(
                    f"  [warn] Post-snap partition constraint violated: "
                    f"word_width={_ps_ww} < partition_product={_ps_partition} "
                    f"(mat_r={_ps_mat_r}, mat_c={_ps_mat_c}, "
                    f"sub_r={_ps_sub_r}, sub_c={_ps_sub_c}). "     
                )

            # Pre-snap: physical values decoded from argmax vocab entries.
            # For Gumbel, argmax is already the discrete selection, so pre_snap == post_snap
            # numerically; reported in the same format as the STE variant for compatibility.
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
            snapped_ppa      = 10 ** pred_snap.cpu().numpy()[0]
            snapped_ppa_dict = {label: snapped_ppa[i] for i, label in enumerate(active_short)}

            pred_ppa = 10 ** best_pred.cpu().numpy()[0]
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
    p = argparse.ArgumentParser(description="DESTINY Inverse Design Optimizer — ST Gumbel-Softmax (minimize)")
    p.add_argument("--tech",                   default="SRAM")
    p.add_argument(
        "--objectives", nargs="+",
        default=["cache_area_mm2",
                "cache_hit_latency_ns",
                "cache_write_latency_ns",
                "cache_hit_energy_nJ",
                "cache_write_energy_nJ",
                "cache_leakage_mW"],
        choices=list(METRIC_META) if "METRIC_META" in dir() else TARGET_KEYS,
        help="Metrics to minimize (log10 sum). e.g. --objectives cache_hit_latency_ns cache_hit_energy_nJ"
    )
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
    p.add_argument(
        "--objective-weights", type=float, nargs="+", default=None,
        metavar="W",
        help="Per-objective weights (one float per --objectives entry). "
             "Higher weight = optimizer pushes harder on that metric. "
             "Controls which region of the Pareto frontier is favoured. "
             "Example: --objective-weights 3.0 1.0  (latency 3x vs energy)"
    )
    p.add_argument(
        "--output", default="runs/results.csv",
        help="CSV to append optimizer results to (default: runs/results.csv)."
    )
    p.add_argument("--verbose-opt", action="store_true",
                   help="Print Gumbel argmax / post-snap parameter table")
    p.add_argument("--model-dir", default=None,
                   help="Custom path to surrogate model directory (contains model.pt, scaler.pkl, feature_cols.json).")
    args = p.parse_args()

    if args.objective_weights is not None and \
            len(args.objective_weights) != len(args.objectives):
        p.error(
            f"--objective-weights must have the same number of entries as --objectives "
            f"(got {len(args.objective_weights)} weights for {len(args.objectives)} objectives)."
        )

    fixed   = dict(args.fix)
    context = build_fixed_context(args.node, args.roadmap, args.temperature, **fixed)

    optimizer = InverseOptimizerGumbelMinimize(args.tech, model_dir=args.model_dir)
    design, ppa, pre_snap, snapped_ppa = optimizer.optimize(
        objectives=args.objectives,
        fixed_context=context,
        steps=args.steps,
        n_restarts=args.restarts,
        objective_weights=args.objective_weights,
        verbose=args.verbose_opt,
    )

    row = {"tech": args.tech, "node_nm": args.node,
           "roadmap": args.roadmap, "temperature_K": args.temperature,
           "capacity_kb": fixed.get("capacity_kb")}
    row.update(design)
    row.update({f"pred_{k}": ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)})
    row.update({f"post_snap_pred_{k}": snapped_ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)})
    row.update({f"objectives": args.objectives})
    row.update({f"pre_snap_{k}": v for k, v in pre_snap.items()})

    df_out = pd.DataFrame([row])
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        already_exists = os.path.exists(args.output)
        df_out.to_csv(args.output, mode="a", index=False, header=not already_exists)
        print(f"Result {'appended to' if already_exists else 'written to'} {args.output}",
              file=sys.stderr)
    else:
        print(df_out.to_csv(index=False), end="")

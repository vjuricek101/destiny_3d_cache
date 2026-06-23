#!/usr/bin/env python3
"""
inverse_design_gumbel_target.py — ST Gumbel-Softmax inverse design optimizer.

Applies Straight-Through Gumbel-Softmax to optimize discrete DESTINY cache configurations
over pre-trained PPA_MLP surrogates under differentiable layout and physical constraints.
"""

from __future__ import annotations

import math
import os
import argparse
import json
import pickle
from typing import Callable

import torch
import numpy as np
import pandas as pd
import sys

from train_model import PPA_MLP, CATEGORICAL_COLS
from destiny_utils import (
    TARGET_COLS as TARGET_KEYS,
    TARGET_SHORT_LABELS,
    get_active_targets,
    build_fixed_context,
    parse_fixed_arg,
    build_gumbel_vocabs,
    build_dyn_idx,
    BASE_ARCH_COLS,
    DATA_PARAM_BOUNDS_LOG2,
    DATA_PARAM_BOUNDS_LINEAR,
    SRAM_CELL_BOUNDS_LOG10,
    SRAM_CELL_BOUNDS_LINEAR,
    A_VTH_BY_NODE,
)
from inverse_physics_validity import (
    compute_physics_penalties,
    check_post_snap_partition,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
GUMBEL_TAU_START: float = 5.0  # Initial temperature (highly uniform)
GUMBEL_TAU_END:   float = 0.5  # Final temperature (near-discrete)
GUMBEL_EPS:       float = 1e-20

ADAM_LR: float = 0.05  # Learning rate for logit optimization

# Loss penalty weights
INFEASIBILITY_PENALTY_WEIGHT: float = 50.0  # Scales surrogate feasibility penalty

# Post-snap defaults & corrections
ASSOC_DEFAULT: int = 4  # Default associativity

# SRAM stability layout constraints: γ-ratio (W_P/W_AC < 0.9) and β-ratio (W_N/W_AC >= 2)
SRAM_WP_WAC_RATIO_CEILING: float = 0.9
SRAM_WN_WAC_RATIO_FLOOR:   float = 2.0
SRAM_WN_MAX_F:             float = 2.5
SRAM_WP_MAX_F:             float = 1.1
SRAM_WAC_MIN_F:            float = 1.0
SRAM_WAC_MAX_F:            float = 1.25

# SRAM cell area model: area = BASE + 30*max(W_N, W_AC) + 20*(W_P + 0.5)
SRAM_CELL_AREA_BASE_F2:     float = 55.0
SRAM_CELL_AREA_NMOS_COEFF:  float = 30.0
SRAM_CELL_AREA_PMOS_COEFF:  float = 20.0
SRAM_CELL_AREA_MIN_F2:      float = 40.0
SRAM_CELL_AREA_MAX_F2:      float = 200.0

# SRAM Pelgrom mismatch: V_sense = 6 * A_Vth / sqrt(2 * W_AC)
SRAM_VSENSE_PELGROM_SIGMA: float = 6.0
SRAM_VSENSE_MIN_MV:        float = 5.0
SRAM_VSENSE_MAX_MV:        float = 80.0
A_VTH_DEFAULT:             float = 3.0

# ── Core Gumbel-Softmax primitive ─────────────────────────────────────────────

def gumbel_softmax_st(logits: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Straight-Through Gumbel-Softmax sample (forward=hard, backward=soft)."""
    u    = torch.rand_like(logits)
    g    = -torch.log(-torch.log(u + GUMBEL_EPS) + GUMBEL_EPS)
    y    = torch.softmax((logits + g) / tau, dim=-1)
    y_hard = torch.zeros_like(logits).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
    y_st   = (y_hard - y).detach() + y
    return y_st, y


# ── Temperature schedule ──────────────────────────────────────────────────────

def _anneal_tau(step: int, total_steps: int) -> float:
    """Exponentially anneal temperature from GUMBEL_TAU_START to GUMBEL_TAU_END."""
    if total_steps <= 1:
        return GUMBEL_TAU_END
    frac    = step / (total_steps - 1)
    log_tau = (1.0 - frac) * math.log(GUMBEL_TAU_START) + frac * math.log(GUMBEL_TAU_END)
    return float(math.exp(log_tau))


# ── Logit initialisation helper ───────────────────────────────────────────────

def _init_logits(size: int, restart: int, device: torch.device) -> torch.Tensor:
    """Initialize parameter logits: zero on restart 0, random noise otherwise."""
    if restart == 0:
        t = torch.zeros(size, device=device)
    else:
        t = torch.randn(size, device=device) * 0.5
    return t.requires_grad_(True)


# ── Snap dispatch table ────────────────────────────────────────────────────────
# Maps each column family to a decoder function: encoded_value -> physical_value.
_SNAP_DECODERS: list[tuple[Callable[[str], bool], Callable[[float], int | float]]] = [
    # capacity_kb: log₁₀ -> physical KB
    (lambda c: c == "capacity_kb",
     lambda v: int(round(10 ** v))),

    # word_width_bits, associativity, stacked die count: log₂ -> physical int
    (lambda c: c in ("word_width_bits", "associativity", "data_stacked_die_count"),
     lambda v: int(round(2 ** v))),

    # MUX levels (mux_sense_amp, mux_output_lev1/2): log₂ -> physical int
    (lambda c: "mux_" in c,
     lambda v: int(round(2 ** v))),

    # Mat counts (num_row_mat, num_col_mat, num_active_mat_per_{row,col}): log₂ -> int
    (lambda c: "_mat_" in c or "num_row_mat" in c or "num_col_mat" in c,
     lambda v: int(round(2 ** v))),

    # Subarray active counts: linear vocab {1.0, 2.0} -> int
    (lambda c: "_subarray_" in c,
     lambda v: int(round(v))),

    # SRAM transistor widths: log₁₀ -> physical width [F]
    (lambda c: c in SRAM_CELL_BOUNDS_LOG10,
     lambda v: float(round(10 ** v, 4))),

    # SRAM linear params (read voltage, aspect ratio): direct physical value
    (lambda c: c in SRAM_CELL_BOUNDS_LINEAR,
     lambda v: float(round(v, 4))),
]


def _snap_design_gumbel(vocabs: dict, logits_dict: dict) -> dict:
    """Argmax logits and decode categorical parameters to physical units."""
    design: dict = {}
    for col, logit in logits_dict.items():
        best_idx = int(logit.detach().argmax().item())
        enc_val  = float(vocabs[col][best_idx].item())

        decoded = enc_val
        for predicate, decoder in _SNAP_DECODERS:
            if predicate(col):
                decoded = decoder(enc_val)
                break

        design[col] = decoded
    return design


# ── SRAM cell feature propagation ─────────────────────────────────────────────

def _update_sram_cell_features(
    x: torch.Tensor,
    dyn_idx: dict,
    encoded_vals_dict: dict,
    fixed_context: dict,
    device: torch.device,
) -> None:
    """Compute and write SRAM cell area and sense voltage features in-place (autograd compatible)."""
    # Decode transistor widths from log₁₀ encoded space → physical [F]
    wn  = 10 ** encoded_vals_dict["CellInput_SRAMCellNMOSWidth (F)"]
    wp  = 10 ** encoded_vals_dict["CellInput_SRAMCellPMOSWidth (F)"]
    wac = 10 ** encoded_vals_dict["CellInput_AccessCMOSWidth (F)"]

    rv_key = "CellInput_ReadVoltage (V)"
    rv = encoded_vals_dict.get(
        rv_key,
        torch.tensor(float(fixed_context.get(rv_key, 1.0)), device=device),
    )

    # Cell area: linearised layout model (mirrors MemCell.cpp)
    cell_area = torch.clamp(
        SRAM_CELL_AREA_BASE_F2
        + SRAM_CELL_AREA_NMOS_COEFF * torch.maximum(wn, wac)
        + SRAM_CELL_AREA_PMOS_COEFF * (wp + 0.5),
        SRAM_CELL_AREA_MIN_F2,
        SRAM_CELL_AREA_MAX_F2,
    )

    if "CellInput_CellArea (F^2)" in dyn_idx:
        x[dyn_idx["CellInput_CellArea (F^2)"]] = torch.log10(cell_area)
    if "derived_sqrt_area" in dyn_idx:
        x[dyn_idx["derived_sqrt_area"]] = torch.sqrt(cell_area)
    if "derived_read_v_sq" in dyn_idx:
        x[dyn_idx["derived_read_v_sq"]] = rv ** 2

    # Sense amp voltage margin: 6σ Pelgrom mismatch at SA input
    if "CellInput_MinSenseVoltage (mV)" in dyn_idx:
        # Resolve process node from fixed_context one-hot encoding
        a_vth = A_VTH_DEFAULT
        for k, v in fixed_context.items():
            if k.startswith("process_node_nm_") and float(v) > 0.5:
                node  = int(k.split("_")[-1])
                a_vth = A_VTH_BY_NODE.get(node, A_VTH_DEFAULT)
                break
        v_sense = torch.clamp(
            SRAM_VSENSE_PELGROM_SIGMA * a_vth / torch.sqrt(2.0 * wac),
            SRAM_VSENSE_MIN_MV,
            SRAM_VSENSE_MAX_MV,
        )
        x[dyn_idx["CellInput_MinSenseVoltage (mV)"]] = v_sense


# ── Main Optimizer class ───────────────────────────────────────────────────────

class InverseOptimizerGumbel:
    """Inverse design optimizer using Straight-Through Gumbel-Softmax to find discrete cache configs."""

    def __init__(self, tech: str) -> None:
        """Load the surrogate model, scaler, and feature metadata for the given technology."""
        self.tech   = tech
        self.device = torch.device("cpu")

        model_dir = f"model_output/{tech.lower()}_feasibility"

        # Feature columns define the input dimension to the surrogate.
        # Loaded from the JSON saved at training time so inference stays
        # consistent with the scaler and weight shapes.
        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            self.feature_cols: list[str] = json.load(f)

        # StandardScaler fit on training data — applied before every forward pass.
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        # Reconstruct PPA_MLP architecture from checkpoint shapes.
        # hidden_dim and n_blocks are inferred so the optimizer stays compatible
        # with any retrained surrogate without requiring a separate config file.
        sd       = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        hidden   = sd["input_proj.weight"].shape[0]
        n_blocks = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
        n_targets = sd["output_head.weight"].shape[0]
        self.model = PPA_MLP(
            len(self.feature_cols),
            hidden_dim=hidden,
            n_blocks=n_blocks,
            n_targets=n_targets,
        ).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        # Active target keys: tech-specific subset of TARGET_KEYS (e.g. SRAM
        # excludes refresh latency/energy which are structurally zero for SRAM).
        self._active_keys: list[str] = get_active_targets(self.tech)

        # Scaler statistics as tensors for efficient in-graph normalisation.
        self.means = torch.tensor(self.scaler.mean_,  dtype=torch.float32, device=self.device)
        self.stds  = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

        # Build the full set of optimisable columns, partitioned by encoding domain.
        # These lists are pruned per-call in optimize() when fixed_context pins some.
        self._all_opt_cols:    list[str] = []
        self._all_log10_cols:  list[str] = []
        self._all_log2_cols:   list[str] = []

        # Base architectural columns: capacity (log₁₀) and the rest (log₂)
        for col in BASE_ARCH_COLS:
            if col in self.feature_cols:
                self._all_opt_cols.append(col)
                if col == "capacity_kb":
                    self._all_log10_cols.append(col)
                else:
                    self._all_log2_cols.append(col)

        # Data / tag array organisational parameters
        for col in DATA_PARAM_BOUNDS_LOG2:
            if col in self.feature_cols:
                self._all_opt_cols.append(col)
                self._all_log2_cols.append(col)

        for col in DATA_PARAM_BOUNDS_LINEAR:
            if col in self.feature_cols:
                self._all_opt_cols.append(col)

        # SRAM-only: transistor width parameters
        if self.tech == "SRAM":
            for col in SRAM_CELL_BOUNDS_LOG10:
                if col in self.feature_cols:
                    self._all_opt_cols.append(col)
                    self._all_log10_cols.append(col)
            for col in SRAM_CELL_BOUNDS_LINEAR:
                if col in self.feature_cols:
                    self._all_opt_cols.append(col)

        # One-hot categorical columns (mem_cell_type, device_roadmap, process_node_nm
        # are pinned by fixed_context and excluded from the vocabulary here).
        self.categorical_vocabs: dict[str, list[str]] = {}
        for cat in CATEGORICAL_COLS:
            if cat in ("mem_cell_type", "device_roadmap", "process_node_nm"):
                continue
            vals = [
                feat[len(cat) + 1:]          # strip "cat_" prefix
                for feat in self.feature_cols
                if feat.startswith(cat + "_")
            ]
            if len(vals) > 1:
                self.categorical_vocabs[cat] = vals

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _target_tensors(
        self, targets: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert target physical metrics to log10 space and return aligned value & mask tensors."""
        vals    = [targets.get(k, 1.0) for k in self._active_keys]
        weights = [1.0 if k in targets else 0.0 for k in self._active_keys]
        return (
            torch.tensor(np.log10(np.clip(vals, 1e-12, None)), dtype=torch.float32, device=self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
        )

    def _build_feature_vector(
        self,
        base_x: torch.Tensor,
        dyn_idx: dict,
        encoded_vals_dict: dict,
        fixed_context: dict,
        cap_enc: torch.Tensor,
        cat_encoded_dict: dict | None = None,
    ) -> torch.Tensor:
        """Construct the complete surrogate input feature vector with dynamic and derived features."""
        x = base_x.clone()

        # 1. Fill continuous optimised parameters directly from encoded values
        for col, enc_val in encoded_vals_dict.items():
            if col in dyn_idx:
                x[dyn_idx[col]] = enc_val

        # 2. Fill soft one-hot categorical features
        if cat_encoded_dict:
            for cat, y_st in cat_encoded_dict.items():
                for i, val in enumerate(self.categorical_vocabs[cat]):
                    col_name = f"{cat}_{val}"
                    if col_name in dyn_idx:
                        x[dyn_idx[col_name]] = y_st[i]

        # 3. Derived cross-parameter features
        cap_phys = 10.0 ** cap_enc  # capacity in KB (physical)

        if "derived_sqrt_capacity" in dyn_idx:
            x[dyn_idx["derived_sqrt_capacity"]] = torch.sqrt(cap_phys)

        die_key  = "data_stacked_die_count"
        stk_phys = (
            2.0 ** encoded_vals_dict[die_key]
            if die_key in encoded_vals_dict
            else torch.tensor(float(fixed_context.get(die_key, 1.0)), device=self.device)
        )

        ww_key  = "word_width_bits"
        ww_phys = (
            2.0 ** encoded_vals_dict[ww_key]
            if ww_key in encoded_vals_dict
            else torch.tensor(2.0 ** float(fixed_context.get(ww_key, 6.0)), device=self.device)
        )

        if "derived_cap_per_die" in dyn_idx:
            x[dyn_idx["derived_cap_per_die"]]  = cap_phys / stk_phys
        if "derived_rows_per_die" in dyn_idx:
            # rows = total_bits / (word_width_bits × die_count)
            x[dyn_idx["derived_rows_per_die"]] = (cap_phys * 1024.0) / (ww_phys * stk_phys)

        # 4. SRAM cell physics (area, sqrt_area, read_v², sense voltage)
        sram_width_cols = set(SRAM_CELL_BOUNDS_LOG10) | set(SRAM_CELL_BOUNDS_LINEAR)
        if self.tech == "SRAM" and sram_width_cols & set(encoded_vals_dict):
            _update_sram_cell_features(x, dyn_idx, encoded_vals_dict, fixed_context, self.device)

        # 5. Roadmap × log₁₀(capacity) interaction term
        # DESTINY's Technology.cpp conditions drive strength on roadmap, so the
        # model sees a capacity-scaled roadmap signal to learn PPA × roadmap coupling.
        for rm in ("HP", "LOP", "LSTP"):
            col = f"device_roadmap_{rm}_x_log10_cap"
            if col in dyn_idx:
                x[dyn_idx[col]] = float(fixed_context.get(f"device_roadmap_{rm}", 0.0)) * cap_enc

        return x

    # ── Main optimisation entry point ─────────────────────────────────────────

    def optimize(
        self,
        targets: dict,
        fixed_context: dict,
        steps: int = 300,
        n_restarts: int = 4,
        target_weights=None,
        verbose: bool = False,
    ) -> tuple[dict, dict, dict, dict]:
        """ST Gumbel-Softmax gradient-based inverse design with random multi-start optimization."""
        # Prune optimisation columns: exclude anything pinned by fixed_context
        fixed_keys       = set(fixed_context.keys())
        self.opt_cols    = [c for c in self._all_opt_cols    if c not in fixed_keys]
        self.log10_cols  = [c for c in self._all_log10_cols  if c not in fixed_keys]
        self.log2_cols   = [c for c in self._all_log2_cols   if c not in fixed_keys]
        self.opt_cats    = [c for c in self.categorical_vocabs if c not in fixed_keys]

        # Build Gumbel vocabularies for this call (may be a subset if some cols fixed)
        vocabs = {k: v.to(self.device) for k, v in build_gumbel_vocabs(self.opt_cols, self.tech).items()}

        t_tensor, w_tensor = self._target_tensors(targets)
        if target_weights is not None:
            if isinstance(target_weights, dict):
                w_tensor = torch.tensor(
                    [target_weights.get(k, 1.0 if k in targets else 0.0) for k in TARGET_KEYS],
                    dtype=torch.float32, device=self.device,
                )
            else:
                w_tensor = torch.tensor(target_weights, dtype=torch.float32, device=self.device)

        # If capacity_kb is fixed, pre-compute its log₁₀ encoded value so that
        # _build_feature_vector can use it consistently throughout all steps.
        _cap_fixed = "capacity_kb" in fixed_keys
        if _cap_fixed:
            _cap_enc_fixed = torch.tensor(
                math.log10(float(fixed_context["capacity_kb"])),
                dtype=torch.float32, device=self.device,
            )

        # dyn_idx maps every feature that changes each step to its index in the
        # flat feature vector, enabling efficient in-place assignment.
        dyn_idx = build_dyn_idx(self.feature_cols, self.opt_cols, self.opt_cats, self.categorical_vocabs)

        # Static base vector: pre-fill all fixed/context features once.
        # Dynamic features are overwritten each step by _build_feature_vector.
        base_x = torch.zeros(len(self.feature_cols), device=self.device)
        for i, c in enumerate(self.feature_cols):
            if c not in dyn_idx:
                base_x[i] = float(fixed_context.get(c, 0.0))

        best_loss, best_logits, best_cat_logits, best_pred = float("inf"), None, None, None

        for restart in range(max(1, n_restarts)):
            # Initialise one logit tensor per optimisable column and category
            logits_dict = {
                col: _init_logits(vocabs[col].shape[0], restart, self.device)
                for col in self.opt_cols
            }
            cat_logits_dict = {
                cat: _init_logits(len(self.categorical_vocabs[cat]), restart, self.device)
                for cat in self.opt_cats
            }

            inner_opt = torch.optim.Adam(
                list(logits_dict.values()) + list(cat_logits_dict.values()),
                lr=ADAM_LR,
            )

            pred = None
            for step in range(steps):
                inner_opt.zero_grad()
                tau = _anneal_tau(step, steps)

                # Sample Gumbel-Softmax for each continuous parameter.
                # The STE inner product (y_st · vocab) gives the differentiable
                # scalar in encoded space (log₁₀, log₂, or linear).
                encoded_vals_dict = {
                    col: (gumbel_softmax_st(logits_dict[col], tau)[0] * vocabs[col]).sum()
                    for col in self.opt_cols
                }

                # Sample Gumbel-Softmax for categorical parameters.
                # y_st is a soft one-hot over the category's values list.
                cat_encoded_dict = {
                    cat: gumbel_softmax_st(cat_logits_dict[cat], tau)[0]
                    for cat in self.opt_cats
                }

                cap_enc = _cap_enc_fixed if _cap_fixed else encoded_vals_dict["capacity_kb"]
                x = self._build_feature_vector(
                    base_x, dyn_idx, encoded_vals_dict,
                    fixed_context, cap_enc, cat_encoded_dict,
                )

                # Normalise and run surrogate forward pass.
                # forward_with_feasibility() returns (log10_ppa [1×T], p_feas [1×1]).
                x_scaled = ((x - self.means) / self.stds).unsqueeze(0)
                pred, p_feas    = self.model.forward_with_feasibility(x_scaled)
                learned_penalty = INFEASIBILITY_PENALTY_WEIGHT * (1.0 - p_feas.squeeze())

                physics_penalty = compute_physics_penalties(
                    encoded_vals_dict, fixed_context, cap_enc, self.device
                )

                # Composite loss: MSE in log₁₀ PPA space + soft constraints
                loss = (w_tensor * (pred - t_tensor) ** 2).sum() + learned_penalty + physics_penalty
                loss.backward()
                inner_opt.step()

            with torch.no_grad():
                final_loss = loss.item()
            if final_loss < best_loss:
                best_loss       = final_loss
                best_logits     = {k: v.detach().clone() for k, v in logits_dict.items()}
                best_cat_logits = {k: v.detach().clone() for k, v in cat_logits_dict.items()}
                best_pred       = pred.detach()

        # ── Post-optimisation: deterministic snap (argmax, no Gumbel noise) ──
        with torch.no_grad():
            # Snap continuous parameters via the dispatch-table decoder
            design = _snap_design_gumbel(vocabs, best_logits)

            # Snap categorical parameters (argmax → string label)
            for cat in self.opt_cats:
                best_idx  = int(best_cat_logits[cat].argmax().item())
                design[cat] = self.categorical_vocabs[cat][best_idx]

            # Associativity was zero-variance in training data; the vocabulary
            # collapses to a single entry so we always pin to the default.
            if "associativity" not in design:
                design["associativity"] = ASSOC_DEFAULT

            # ── SRAM transistor ratio post-correction ────────────────────────
            # After snapping, the discrete transistor widths may violate the
            # γ-ratio (W_P/W_AC < 1) or β-ratio (W_N/W_AC ≥ 2) stability
            # requirements.  We apply minimal corrections to repair violations
            # before passing the design to DESTINY.
            if self.tech == "SRAM" and "CellInput_SRAMCellNMOSWidth (F)" in design:
                wn  = design["CellInput_SRAMCellNMOSWidth (F)"]
                wp  = design["CellInput_SRAMCellPMOSWidth (F)"]
                wac = design["CellInput_AccessCMOSWidth (F)"]

                # γ-ratio check: W_P must be strictly less than W_AC
                if wp / wac >= 1.0:
                    if SRAM_WP_WAC_RATIO_CEILING * wac > SRAM_WAC_MIN_F:
                        wp = float(round(SRAM_WP_WAC_RATIO_CEILING * wac, 2))
                    else:
                        # wac is too small; bump wac up to allow a valid wp
                        wac = float(round(min(SRAM_WAC_MIN_F / SRAM_WP_WAC_RATIO_CEILING + 0.01, SRAM_WAC_MAX_F), 2))
                        wp  = float(round(min(SRAM_WP_WAC_RATIO_CEILING * wac, SRAM_WP_MAX_F), 2))
                    design["CellInput_AccessCMOSWidth (F)"]   = wac
                    design["CellInput_SRAMCellPMOSWidth (F)"] = wp

                # β-ratio check: W_N must be at least 2× W_AC
                if wn / wac < SRAM_WN_WAC_RATIO_FLOOR:
                    design["CellInput_SRAMCellNMOSWidth (F)"] = float(
                        round(min(SRAM_WN_WAC_RATIO_FLOOR * wac, SRAM_WN_MAX_F), 2)
                    )

            # Merge fixed_context values not covered by the optimised columns
            for k, v in fixed_context.items():
                if k not in self.opt_cols:
                    design[k] = v

            # Warn if the snapped design violates any DESTINY partition constraint
            # (mirrors the DESTINY main.cpp check that silently discards configs).
            check_post_snap_partition(design, fixed_context)

            # ── Pre-snap: argmax vocab → physical (no rounding corrections) ──
            # For the Gumbel variant, pre_snap and post_snap are numerically
            # identical (argmax is already discrete); reported for API parity
            # with the STE variant which has a continuous pre_snap.
            pre_snap: dict = {}
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
                best_idx     = int(best_cat_logits[cat].argmax().item())
                pre_snap[cat] = self.categorical_vocabs[cat][best_idx]

            # ── Second forward pass on snapped design → honest post-snap PPA ─
            # Re-encode the snapped (discrete) values and run a clean forward
            # pass.  This gives the PPA the surrogate would predict for the
            # exact design that would be submitted to DESTINY — no Gumbel noise,
            # no temperature relaxation, no gradient tracking.
            snapped_encoded = {
                col: torch.tensor(
                    float(vocabs[col][int(best_logits[col].argmax().item())].item()),
                    device=self.device,
                )
                for col in self.opt_cols
            }
            snapped_cat_encoded = {
                cat: torch.zeros(len(self.categorical_vocabs[cat]), device=self.device).scatter_(
                    0,
                    torch.tensor(int(best_cat_logits[cat].argmax().item()), device=self.device),
                    1.0,
                )
                for cat in self.opt_cats
            }

            cap_enc_snap = _cap_enc_fixed if _cap_fixed else snapped_encoded["capacity_kb"]
            x_snap       = self._build_feature_vector(
                base_x, dyn_idx, snapped_encoded,
                fixed_context, cap_enc_snap, snapped_cat_encoded,
            )
            x_snap_scaled = ((x_snap - self.means) / self.stds).unsqueeze(0)
            pred_snap, _  = self.model.forward_with_feasibility(x_snap_scaled)

            # Decode predictions back to physical units (surrogate outputs log₁₀)
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
    # Sanity-check that TARGET_KEYS and TARGET_SHORT_LABELS stay aligned.
    # If these ever diverge (e.g. after editing destiny_utils.py) the row
    # assembly below would silently produce mismatched columns in the CSV.
    assert len(TARGET_KEYS) == len(TARGET_SHORT_LABELS), (
        f"TARGET_KEYS ({len(TARGET_KEYS)}) and TARGET_SHORT_LABELS "
        f"({len(TARGET_SHORT_LABELS)}) must have the same length."
    )

    p = argparse.ArgumentParser(description="DESTINY Inverse Design Optimizer — ST Gumbel-Softmax")
    p.add_argument("--tech",                   default="SRAM")
    p.add_argument("--target-read-latency",    type=float, help="Target read latency (ns)")
    p.add_argument("--target-write-latency",   type=float, help="Target write latency (ns)")
    p.add_argument("--target-refresh-latency", type=float, help="Target refresh latency (ns)")
    p.add_argument("--target-area",            type=float, help="Target cache area (mm²)")
    p.add_argument("--target-hit-energy",      type=float, help="Target hit energy (nJ)")
    p.add_argument("--target-write-energy",    type=float, help="Target write energy (nJ)")
    p.add_argument("--target-refresh-energy",  type=float, help="Target refresh energy (nJ)")
    p.add_argument("--target-leakage",         type=float, help="Target leakage power (mW)")
    p.add_argument("--node",        type=int, default=32, choices=[22, 32, 45, 65])
    p.add_argument("--roadmap",     default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--temperature", type=float, default=350.0)
    p.add_argument("--steps",       type=int,   default=300)
    p.add_argument("--restarts",    type=int,   default=4)
    p.add_argument(
        "--fix", nargs="*", default=[], metavar="KEY=VALUE", type=parse_fixed_arg,
        help=(
            "Pin design parameters as optimizer constants. "
            "Values are auto-coerced to int, float, or str. "
            "E.g. --fix capacity_kb=64 associativity=8"
        ),
    )
    p.add_argument("--output",      default=None, help="CSV to append results to")
    p.add_argument("--verbose-opt", action="store_true",
                   help="Print Gumbel argmax / post-snap parameter table")
    args = p.parse_args()

    # Build target dict: only include metrics that were explicitly specified
    targets = {
        key: val
        for key, val in [
            ("cache_area_mm2",           args.target_area),
            ("cache_hit_latency_ns",     args.target_read_latency),
            ("cache_write_latency_ns",   args.target_write_latency),
            ("cache_refresh_latency_ns", args.target_refresh_latency),
            ("cache_hit_energy_nJ",      args.target_hit_energy),
            ("cache_write_energy_nJ",    args.target_write_energy),
            ("cache_refresh_energy_nJ",  args.target_refresh_energy),
            ("cache_leakage_mW",         args.target_leakage),
        ]
        if val is not None
    }

    if not targets:
        p.error("Specify at least one target via --target-read-latency / --target-area / etc.")

    fixed   = dict(args.fix)
    context = build_fixed_context(args.node, args.roadmap, args.temperature, **fixed)

    optimizer = InverseOptimizerGumbel(args.tech)
    design, ppa, pre_snap, snapped_ppa = optimizer.optimize(
        targets, context,
        steps=args.steps,
        n_restarts=args.restarts,
        verbose=args.verbose_opt,
    )

    # Assemble output row: design parameters + pre-snap values + predicted PPA
    # + post-snap PPA + targets (NaN for unspecified metrics).
    row: dict = {"tech": args.tech, "node_nm": args.node,
                 "roadmap": args.roadmap, "temperature_K": args.temperature}
    row.update(design)
    row.update({f"pred_{k}":          ppa.get(label)         for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)})
    row.update({f"post_snap_pred_{k}": snapped_ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_SHORT_LABELS)})
    row.update({f"target_{k}":        targets.get(k, float("nan")) for k in TARGET_KEYS})
    row.update({f"pre_snap_{k}":      v for k, v in pre_snap.items()})

    df_out = pd.DataFrame([row])
    if args.output:
        df_out.to_csv(args.output, mode="a", index=False,
                      header=not os.path.exists(args.output))
        print(f"Result appended to {args.output}", file=sys.stderr)
    else:
        print(df_out.to_csv(index=False), end="")

#!/usr/bin/env python3
"""
inverse_design_gumbel.py
========================
During the inner optimisation loop we:
  1. Sample i.i.d. Gumbel noise:  g = -log(-log(U + ε) + ε),  U ~ Uniform(0,1)
  2. Compute the continuous Gumbel-Softmax relaxation:
        y_i = softmax( (logits_i + g_i) / τ )
  3. Apply the Straight-Through (ST) trick so the forward pass uses a hard
     argmax selection while gradients flow back through the soft relaxation y:
        y_hard = one_hot( argmax(y) )
        y_st   = (y_hard - y).detach() + y
  4.  The scalar value fed into the surrogate model is obtained as the inner
      product of y_st with the vocabulary tensor, which is differentiable w.r.t.
      the logits in the backward pass.

Temperature annealing
---------------------
τ is decayed exponentially from GUMBEL_TAU_START → GUMBEL_TAU_END over the
optimisation steps so that the distribution hardens progressively.
"""

import math
import os
import argparse
import json
import pickle

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
    TARGET_KEY_TO_OPT_TARGET
)

# ---------------------------------------------------------------------------
# Shared constants (identical to inverse_design.py)
# ---------------------------------------------------------------------------

_MUX_VALID = [1, 2, 4, 8, 16, 32, 64]
_MAT_VALID = [1, 2, 4, 8, 16]

# Continuous-space bounds (kept for SRAM cell feature derivation helpers)
BASE_ARCH_COLS   = ["capacity_kb", "word_width_bits", "associativity", "data_stacked_die_count"]
BASE_ARCH_BOUNDS = [
    (np.log10(2),  np.log10(32768)),  # capacity_kb  (log10)
    (6,            11),               # word_width    (log2)
    (0,            6),                # associativity (log2)
    (0,            4),                # stacked_die   (log2)
]

DATA_PARAM_BOUNDS_LOG2 = {
    "data_mux_sense_amp":          (0, 6),
    "data_mux_output_lev2":        (0, 6),
    "data_num_active_mat_per_row": (0, 4),
    "data_num_active_mat_per_col": (0, 4),
}
DATA_PARAM_BOUNDS_LINEAR = {
    "data_num_active_subarray_per_row": (1, 2),
    "data_num_active_subarray_per_col": (1, 2),
}
SRAM_CELL_BOUNDS_LOG10 = {
    "CellInput_SRAMCellNMOSWidth (F)": (2.2, 2.5),
    "CellInput_SRAMCellPMOSWidth (F)": (1.0, 1.1),
    "CellInput_AccessCMOSWidth (F)":   (1.1, 1.25),
}
SRAM_CELL_BOUNDS_LINEAR = {
    "CellInput_ReadVoltage (V)": (0.5, 1.2),
}

# ---------------------------------------------------------------------------
# Gumbel-Softmax hyper-parameters
# ---------------------------------------------------------------------------

GUMBEL_TAU_START = 5.0   # hot  → broad distribution, low gradient variance
GUMBEL_TAU_END   = 0.5   # cold → near-discrete, higher variance

# ---------------------------------------------------------------------------
# Categorical vocabulary definitions
# ---------------------------------------------------------------------------

# All vocabulary values are stored in the *encoded* space that the feature
# vector expects:
#   • log10-encoded parameters → stored as log10(physical_value)
#   • log2-encoded parameters  → stored as log2(physical_value)
#   • linear parameters        → stored as the physical value directly

# Shared helpers
_CAP_KB_PHYS   = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
_WW_PHYS       = [64, 128, 256, 512, 1024, 2048]       # 2^6 … 2^11
_ASSOC_PHYS    = [1, 2, 4, 8, 16, 32, 64]              # 2^0 … 2^6
_DIE_PHYS      = [1, 2, 4, 8, 16]                      # 2^0 … 2^4


def _log10_vocab(phys_values):
    """Return a float32 tensor of log10-encoded vocabulary entries."""
    return torch.tensor([math.log10(v) for v in phys_values], dtype=torch.float32)


def _log2_vocab(phys_values):
    """Return a float32 tensor of log2-encoded vocabulary entries."""
    return torch.tensor([math.log2(v) for v in phys_values], dtype=torch.float32)


def _linear_grid(lo, hi, step):
    """Return a float32 tensor of linearly-spaced values [lo, lo+step, …, hi]."""
    n = max(2, round((hi - lo) / step) + 1)
    vals = [lo + i * step for i in range(n)]
    # clamp last entry to hi to avoid floating-point overshoot
    vals[-1] = hi
    return torch.tensor(vals, dtype=torch.float32)


def _build_gumbel_vocabs(opt_cols: list, tech: str) -> dict:
    """
    Build an ordered dict mapping each column name in *opt_cols* to a 1-D
    float32 tensor of discrete vocabulary values **in encoded space**.

    Parameters
    ----------
    opt_cols : list[str]
        The subset of columns that will be optimised (fixed_context columns
        have already been removed).
    tech : str
        Technology node string, e.g. ``"SRAM"``.

    Returns
    -------
    dict[str, torch.Tensor]
        Ordered dict (same order as opt_cols) with one vocabulary tensor per
        column.
    """
    vocabs = {}
    for col in opt_cols:
        if col == "capacity_kb":
            vocabs[col] = _log10_vocab(_CAP_KB_PHYS)

        elif col == "word_width_bits":
            vocabs[col] = _log2_vocab(_WW_PHYS)

        elif col == "associativity":
            vocabs[col] = _log2_vocab(_ASSOC_PHYS)

        elif col == "data_stacked_die_count":
            vocabs[col] = _log2_vocab(_DIE_PHYS)

        elif col in ("data_mux_sense_amp", "data_mux_output_lev2"):
            # MUX factor — stored as log2(physical) in the feature vector
            vocabs[col] = _log2_vocab(_MUX_VALID)

        elif col in ("data_num_active_mat_per_row", "data_num_active_mat_per_col"):
            # Mat count — stored as log2(physical)
            vocabs[col] = _log2_vocab(_MAT_VALID)

        elif col in ("data_num_active_subarray_per_row", "data_num_active_subarray_per_col"):
            # Integer linear — valid values are just {1, 2}
            vocabs[col] = torch.tensor([1.0, 2.0], dtype=torch.float32)

        elif tech == "SRAM" and col in SRAM_CELL_BOUNDS_LOG10:
            # SRAM transistor widths — fine 0.01-step grid in log10 space
            lo_phys, hi_phys = SRAM_CELL_BOUNDS_LOG10[col]
            lo_enc = math.log10(lo_phys)
            hi_enc = math.log10(hi_phys)
            vocabs[col] = _linear_grid(lo_enc, hi_enc, step=0.01)

        elif tech == "SRAM" and col in SRAM_CELL_BOUNDS_LINEAR:
            # Read voltage — fine 0.05-step grid in linear space
            lo, hi = SRAM_CELL_BOUNDS_LINEAR[col]
            vocabs[col] = _linear_grid(lo, hi, step=0.05)

        else:
            # Fallback: treat as a 2-element {0, 1} binary vocabulary so the
            # rest of the code never KeyErrors.  In practice this branch should
            # not be reached for well-known columns.
            vocabs[col] = torch.tensor([0.0, 1.0], dtype=torch.float32)

    return vocabs


# ---------------------------------------------------------------------------
# Core Gumbel-Softmax primitive
# ---------------------------------------------------------------------------

def gumbel_softmax_st(logits: torch.Tensor, tau: float):
    """
    Straight-Through Gumbel-Softmax sample.

    Implements Equations (1)–(4) from Jang et al., ICLR 2017.

    Forward pass  : hard one-hot via argmax (exact discrete selection).
    Backward pass : gradients flow through the soft Gumbel-Softmax relaxation y.

    Parameters
    ----------
    logits : torch.Tensor, shape [K]
        Unnormalised log-probabilities (log π_i) for K discrete categories.
    tau : float
        Current temperature τ > 0.

    Returns
    -------
    y_st : torch.Tensor, shape [K]
        STE one-hot vector — has gradient w.r.t. logits.
    y    : torch.Tensor, shape [K]
        Soft Gumbel-Softmax sample — used for gradient computation only.
    """
    # Sample i.i.d. Gumbel(0,1) noise via inverse transform sampling
    # g = -log(-log(U + ε) + ε),  U ~ Uniform(0,1)
    eps = 1e-20
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)

    # Continuous Gumbel-Softmax relaxation (soft sample on the simplex)
    y = torch.softmax((logits + g) / tau, dim=-1)

    # Hard selection — one-hot at argmax position — used in forward pass
    y_hard = torch.zeros_like(logits).scatter_(
        -1, y.argmax(-1, keepdim=True), 1.0
    )

    # Straight-Through estimator:
    # y_st == y_hard in the forward pass
    # ∇_logits y_st ≈ ∇_logits y   (gradient passes through y unchanged)
    y_st = (y_hard - y).detach() + y

    return y_st, y


# ---------------------------------------------------------------------------
# Temperature annealing schedule
# ---------------------------------------------------------------------------

def _anneal_tau(step: int, total_steps: int) -> float:
    """
    Exponential temperature annealing from GUMBEL_TAU_START → GUMBEL_TAU_END.

    Interpolates linearly in log-space so that τ == GUMBEL_TAU_START at step 0
    and τ == GUMBEL_TAU_END exactly at step == total_steps - 1.
    """
    if total_steps <= 1:
        return GUMBEL_TAU_END
    frac = step / (total_steps - 1)          # 0.0 → 1.0
    log_tau = (1.0 - frac) * math.log(GUMBEL_TAU_START) + frac * math.log(GUMBEL_TAU_END)
    return float(math.exp(log_tau))


# ---------------------------------------------------------------------------
# Module-level helpers (carried over from inverse_design.py)
# ---------------------------------------------------------------------------

def _nearest_valid(val: int, valid_set: list) -> int:
    return min(valid_set, key=lambda v: abs(v - val))


def _select_opt_target_col(targets, fixed_context):
    """Return the opt_target one-hot column name matching the primary target metric."""
    explicit = fixed_context.get("_opt_target")
    if explicit:
        return f"opt_target_{explicit}"
    primary = next(iter(targets.keys()), "cache_hit_latency_ns")
    return f"opt_target_{TARGET_KEY_TO_OPT_TARGET.get(primary, 'Read Latency')}"


def _build_dyn_idx(feature_cols, opt_cols):
    """Index of all columns updated per gradient step."""
    dyn_keys = set(opt_cols) | {
        "derived_sqrt_capacity", "derived_cap_per_die", "derived_rows_per_die",
        "CellInput_CellArea (F^2)", "derived_sqrt_area", "derived_read_v_sq",
        "CellInput_MinSenseVoltage (mV)", "CellInput_CellAspectRatio",
    } | {f"device_roadmap_{rm}_x_log10_cap" for rm in ["HP", "LOP", "LSTP"]}
    return {k: i for i, k in enumerate(feature_cols) if k in dyn_keys}


def _update_sram_cell_features(x, dyn_idx, encoded_vals_dict, fixed_context, device):
    """
    Compute SRAM cell derived features and write them into feature vector x
    (in-place).  Accepts *encoded_vals_dict* — a dict {col: scalar_tensor_in_encoded_space}.

    This is the Gumbel variant: values arrive as differentiable scalars (inner
    products of one-hot y_st and the vocab tensor), so autograd graphs are intact.
    """
    wn  = 10 ** encoded_vals_dict["CellInput_SRAMCellNMOSWidth (F)"]
    wp  = 10 ** encoded_vals_dict["CellInput_SRAMCellPMOSWidth (F)"]
    wac = 10 ** encoded_vals_dict["CellInput_AccessCMOSWidth (F)"]

    rv_key = "CellInput_ReadVoltage (V)"
    if rv_key in encoded_vals_dict:
        rv = encoded_vals_dict[rv_key]          # linear — already physical
    else:
        rv = torch.tensor(float(fixed_context.get(rv_key, 1.0)), device=device)

    # Differentiable CellArea — gradient flows back through y_st → logits
    cell_area = 55.0 + 30.0 * torch.maximum(wn, wac) + 20.0 * (wp + 0.5)
    cell_area = torch.clamp(cell_area, 40.0, 200.0)

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
        v_sense = 6.0 * a_vth / torch.sqrt(2.0 * wac)
        v_sense = torch.clamp(v_sense, 5.0, 80.0)
        x[dyn_idx["CellInput_MinSenseVoltage (mV)"]] = v_sense

    if "CellInput_CellAspectRatio" in dyn_idx:
        x[dyn_idx["CellInput_CellAspectRatio"]] = torch.tensor(1.4600, device=device)


# ---------------------------------------------------------------------------
# Post-optimisation snap (deterministic — no Gumbel noise)
# ---------------------------------------------------------------------------

def _snap_design_gumbel(vocabs: dict, logits_dict: dict) -> dict:
    """
    Convert final logits to physical design values by taking the argmax of
    each logit vector and looking up the corresponding vocabulary entry.

    Parameters
    ----------
    vocabs      : {col: vocab_tensor_in_encoded_space}
    logits_dict : {col: logit_tensor}

    Returns
    -------
    design : {col: physical_value (python scalar)}
        Values are decoded back to *physical* space (not encoded space).
    """
    design = {}
    for col, logit in logits_dict.items():
        vocab      = vocabs[col]
        best_idx   = int(logit.detach().argmax().item())
        enc_val    = float(vocab[best_idx].item())

        # Decode encoded → physical
        if col == "capacity_kb":
            design[col] = int(round(10 ** enc_val))
        elif col in ("word_width_bits", "associativity", "data_stacked_die_count"):
            design[col] = int(round(2 ** enc_val))
        elif col in ("data_mux_sense_amp", "data_mux_output_lev2"):
            design[col] = int(round(2 ** enc_val))
        elif col in ("data_num_active_mat_per_row", "data_num_active_mat_per_col"):
            design[col] = int(round(2 ** enc_val))
        elif col in ("data_num_active_subarray_per_row", "data_num_active_subarray_per_col"):
            design[col] = int(round(enc_val))
        elif col in SRAM_CELL_BOUNDS_LOG10:
            design[col] = float(round(10 ** enc_val, 4))
        elif col in SRAM_CELL_BOUNDS_LINEAR:
            design[col] = float(round(enc_val, 4))
        else:
            design[col] = enc_val

    return design


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------

class InverseOptimizerGumbel:
    """
    Inverse design optimiser using Straight-Through Gumbel-Softmax.

    Each architectural parameter is modelled as a categorical variable over
    an explicit discrete vocabulary.  Gradient descent updates a set of
    unconstrained logit tensors; Gumbel-Softmax with temperature annealing
    provides differentiable samples throughout.

    The public interface is identical to InverseOptimizer in inverse_design.py:
    ``optimize()`` returns ``(design, ppa_dict, pre_snap, snapped_ppa_dict)``.
    """

    def __init__(self, tech: str, use_feasibility: bool = False):
        self.tech   = tech
        self.device = torch.device("cpu")

        if use_feasibility:
            model_dir = f"model_output/{tech.lower()}_feasibility"
        else:
            model_dir = f"model_output/{tech.lower()}_full_with_data_params"
            if not os.path.exists(model_dir):
                model_dir = f"model_output/{tech.lower()}_full"
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
        has_feas = "feasibility_head.weight" in sd
        self.model = PPA_MLP(len(self.feature_cols), hidden_dim=hidden,
                             n_blocks=n_blocks, has_feasibility_head=has_feas).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        self.means = torch.tensor(self.scaler.mean_,  dtype=torch.float32, device=self.device)
        self.stds  = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)

        # Candidate columns — pruned per-call in optimize() based on fixed_context
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

    # ------------------------------------------------------------------

    def _target_tensors(self, targets):
        """Return (log10 target values, binary weight mask) for the 4 PPA outputs."""
        vals    = [targets.get(k, 1.0) for k in TARGET_KEYS]
        weights = [1.0 if k in targets else 0.0 for k in TARGET_KEYS]
        return (
            torch.tensor(np.log10(np.clip(vals, 1e-12, None)), dtype=torch.float32, device=self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
        )

    # ------------------------------------------------------------------

    def _build_feature_vector(
        self,
        base_x: torch.Tensor,
        dyn_idx: dict,
        encoded_vals_dict: dict,
        opt_cols: list,
        fixed_context: dict,
        cap_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct the full feature vector for the surrogate model.

        Parameters
        ----------
        base_x            : static feature vector (fixed_context values)
        dyn_idx           : {col_name: feature_index}
        encoded_vals_dict : {col: scalar tensor in encoded space, differentiable}
        opt_cols          : ordered list of optimised column names
        fixed_context     : raw context dict
        cap_enc           : log10(capacity_kb) as a scalar tensor

        Returns
        -------
        x : feature vector (1-D tensor), differentiable w.r.t. logits
        """
        x = base_x.clone()

        # Write each optimised parameter into the feature vector
        for col, enc_val in encoded_vals_dict.items():
            if col in dyn_idx:
                x[dyn_idx[col]] = enc_val

        # derived_sqrt_capacity needs physical capacity
        cap_phys = 10.0 ** cap_enc
        if "derived_sqrt_capacity" in dyn_idx:
            x[dyn_idx["derived_sqrt_capacity"]] = torch.sqrt(cap_phys)

        # Stacked die count
        die_key = "data_stacked_die_count"
        if die_key in encoded_vals_dict:
            stk_phys = 2.0 ** encoded_vals_dict[die_key]
        else:
            stk_phys = torch.tensor(
                float(fixed_context.get(die_key, 1.0)), device=self.device
            )

        # Word width
        ww_key = "word_width_bits"
        if ww_key in encoded_vals_dict:
            ww_phys = 2.0 ** encoded_vals_dict[ww_key]
        else:
            ww_phys = torch.tensor(
                2.0 ** float(fixed_context.get(ww_key, 6.0)), device=self.device
            )

        if "derived_cap_per_die" in dyn_idx:
            x[dyn_idx["derived_cap_per_die"]]  = cap_phys / stk_phys
        if "derived_rows_per_die" in dyn_idx:
            x[dyn_idx["derived_rows_per_die"]] = (cap_phys * 1024.0) / (ww_phys * stk_phys)

        # SRAM cell features
        sram_width_cols = set(SRAM_CELL_BOUNDS_LOG10) | set(SRAM_CELL_BOUNDS_LINEAR)
        if self.tech == "SRAM" and sram_width_cols & set(encoded_vals_dict):
            _update_sram_cell_features(x, dyn_idx, encoded_vals_dict, fixed_context, self.device)

        # Roadmap × log10_cap interaction features
        for rm in ["HP", "LOP", "LSTP"]:
            col = f"device_roadmap_{rm}_x_log10_cap"
            if col in dyn_idx:
                x[dyn_idx[col]] = float(fixed_context.get(f"device_roadmap_{rm}", 0.0)) * cap_enc

        return x

    # ------------------------------------------------------------------

    def optimize(
        self,
        targets: dict,
        fixed_context: dict,
        steps: int = 300,
        n_restarts: int = 4,
        target_weights=None,
        verbose: bool = False,
    ):
        """
        ST Gumbel-Softmax gradient-based inverse design with multi-start.

        Parameters
        ----------
        targets        : {metric_key: target_physical_value}
        fixed_context  : {col: value} — columns pinned and not optimised
        steps          : gradient steps per restart
        n_restarts     : number of random restarts
        target_weights : optional override for per-metric loss weights
        verbose        : if True, print per-parameter table at the end

        Returns
        -------
        design         : {col: physical_value} — snapped hardware design
        ppa_dict       : {label: predicted_value} — continuous-param PPA
        pre_snap       : {col: pre-snap physical value}
        snapped_ppa_dict : {label: predicted_value} — post-snap PPA
        """
        # Prune fixed columns
        fixed_keys       = set(fixed_context.keys())
        self.opt_cols    = [c for c in self._all_opt_cols    if c not in fixed_keys]
        self.log10_cols  = [c for c in self._all_log10_cols  if c not in fixed_keys]
        self.log2_cols   = [c for c in self._all_log2_cols   if c not in fixed_keys]
        self.linear_cols = [c for c in self._all_linear_cols if c not in fixed_keys]

        # Build categorical vocabularies (encoded space, on CPU)
        vocabs = _build_gumbel_vocabs(self.opt_cols, self.tech)
        # Move to device
        vocabs = {k: v.to(self.device) for k, v in vocabs.items()}

        t_tensor, w_tensor = self._target_tensors(targets)
        if target_weights is not None:
            if isinstance(target_weights, dict):
                w_tensor = torch.tensor(
                    [target_weights.get(k, 1.0 if k in targets else 0.0) for k in TARGET_KEYS],
                    dtype=torch.float32, device=self.device)
            else:
                w_tensor = torch.tensor(target_weights, dtype=torch.float32, device=self.device)

        dyn_idx = _build_dyn_idx(self.feature_cols, self.opt_cols)

        # Static base feature vector from fixed_context
        base_x = torch.zeros(len(self.feature_cols), device=self.device)
        for i, c in enumerate(self.feature_cols):
            if c not in dyn_idx:
                base_x[i] = float(fixed_context.get(c, 0.0))

        # Condition opt_target one-hot
        for i, c in enumerate(self.feature_cols):
            if c.startswith("opt_target_"):
                base_x[i] = 1.0 if c == _select_opt_target_col(targets, fixed_context) else 0.0

        best_loss, best_logits, best_pred = float("inf"), None, None

        for restart in range(max(1, n_restarts)):
            # ----------------------------------------------------------------
            # Initialise logit tensors
            # Each logit vector starts as zeros (uniform categorical prior)
            # for restart 0, and as small random perturbations thereafter.
            # ----------------------------------------------------------------
            logits_dict = {}
            for col in self.opt_cols:
                K = vocabs[col].shape[0]
                if restart == 0:
                    init = torch.zeros(K, device=self.device)
                else:
                    init = torch.randn(K, device=self.device) * 0.5
                logits_dict[col] = init.requires_grad_(True)

            params_list = list(logits_dict.values())
            inner_opt   = torch.optim.Adam(params_list, lr=0.05)

            pred = None  # populated inside the loop

            for step in range(steps):
                inner_opt.zero_grad()

                tau = _anneal_tau(step, steps)

                # ---- Sample from Gumbel-Softmax for each parameter --------
                encoded_vals_dict = {}  # col → differentiable scalar in encoded space
                for col in self.opt_cols:
                    logit  = logits_dict[col]
                    vocab  = vocabs[col]
                    y_st, _y = gumbel_softmax_st(logit, tau)
                    # Scalar encoded value: inner product of STE one-hot and vocab
                    # Forward: exact hard value.  Backward: gradient through y_st.
                    enc_val = (y_st * vocab).sum()
                    encoded_vals_dict[col] = enc_val

                cap_enc = encoded_vals_dict["capacity_kb"]

                # ---- Build feature vector ---------------------------------
                x = self._build_feature_vector(
                    base_x, dyn_idx, encoded_vals_dict, self.opt_cols, fixed_context, cap_enc
                )

                # ---- Surrogate model forward pass -------------------------
                x_scaled = ((x - self.means) / self.stds).unsqueeze(0)
                if self.model.has_feasibility_head:
                    pred, p_feas    = self.model.forward_with_feasibility(x_scaled)
                    learned_penalty = 50.0 * (1.0 - p_feas.squeeze())
                else:
                    pred            = self.model(x_scaled)
                    learned_penalty = 0.0

                # ---- Loss -------------------------------------------------
                loss = (w_tensor * (pred - t_tensor) ** 2).sum() + learned_penalty
                loss.backward()
                inner_opt.step()

            with torch.no_grad():
                final_loss = loss.item()
            if final_loss < best_loss:
                best_loss   = final_loss
                best_logits = {k: v.detach().clone() for k, v in logits_dict.items()}
                best_pred   = pred.detach()

        # ------------------------------------------------------------------
        # Post-optimisation: deterministic snap (argmax, no Gumbel noise)
        # ------------------------------------------------------------------
        with torch.no_grad():
            design = _snap_design_gumbel(vocabs, best_logits)

            # Associativity default (zero-variance in training sweep)
            if "associativity" not in design:
                design["associativity"] = 4

            # Enforce SRAM transistor ratio constraints
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

            # Merge fixed_context values not already in design
            for k, v in fixed_context.items():
                if k not in self.opt_cols:
                    design[k] = v

            # ---- Pre-snap: physical values from argmax encoded vocabulary ----
            # (For Gumbel, "pre-snap" = what the argmax vocab entry decodes to,
            # which is already the exact discrete value; we report it in the
            # same format as the continuous case for drop-in compatibility.)
            pre_snap = {}
            for col in self.opt_cols:
                vocab    = vocabs[col]
                best_idx = int(best_logits[col].argmax().item())
                enc_val  = float(vocab[best_idx].item())
                if col in self.log10_cols:
                    pre_snap[col] = float(10 ** enc_val)
                elif col in self.log2_cols:
                    pre_snap[col] = float(2 ** enc_val)
                else:
                    pre_snap[col] = enc_val

            # ---- Second forward pass with snapped values (honest prediction) --
            snapped_encoded = {}
            for col in self.opt_cols:
                vocab    = vocabs[col]
                best_idx = int(best_logits[col].argmax().item())
                snapped_encoded[col] = torch.tensor(float(vocab[best_idx].item()), device=self.device)

            cap_enc_snap = snapped_encoded["capacity_kb"]
            x_snap = self._build_feature_vector(
                base_x, dyn_idx, snapped_encoded, self.opt_cols, fixed_context, cap_enc_snap
            )
            x_snap_scaled = ((x_snap - self.means) / self.stds).unsqueeze(0)
            if self.model.has_feasibility_head:
                pred_snap, _ = self.model.forward_with_feasibility(x_snap_scaled)
            else:
                pred_snap = self.model(x_snap_scaled)

            snapped_ppa     = 10 ** pred_snap.cpu().numpy()[0]
            snapped_ppa_dict = {label: snapped_ppa[i] for i, label in enumerate(TARGET_LABELS)}

            pred_ppa = 10 ** best_pred.cpu().numpy()[0]
            ppa_dict = {label: pred_ppa[i] for i, label in enumerate(TARGET_LABELS)}

            if verbose:
                col_w = max(len(c) for c in self.opt_cols) + 2
                print(f"\n  {'Parameter':<{col_w}}  {'Gumbel argmax (physical)':>25}  {'Post-snap (physical)':>20}")
                print(f"  {'-'*col_w}  {'':->25}  {'':->20}")
                for col in self.opt_cols:
                    pre  = pre_snap.get(col, float("nan"))
                    post = design.get(col, float("nan"))
                    print(f"  {col:<{col_w}}  {pre:>25.6g}  {post:>20.6g}")
                print()

        return design, ppa_dict, pre_snap, snapped_ppa_dict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="DESTINY Inverse Design Optimizer — ST Gumbel-Softmax variant"
    )
    p.add_argument("--tech",           default="SRAM")
    p.add_argument("--target-read-latency",    type=float, help="Target read latency (ns)")
    p.add_argument("--target-write-latency",   type=float, help="Target write latency (ns)")
    p.add_argument("--target-refresh-latency", type=float, help="Target refresh latency (ns)")
    p.add_argument("--target-area",            type=float, help="Target cache area (mm2)")
    p.add_argument("--target-hit-energy",      type=float, help="Target hit energy (nJ)")
    p.add_argument("--target-write-energy",    type=float, help="Target write energy (nJ)")
    p.add_argument("--target-refresh-energy",  type=float, help="Target refresh energy (nJ)")
    p.add_argument("--target-leakage",         type=float, help="Target leakage power (mW)")
    # Keep legacy names for CLI backward compatibility:
    p.add_argument("--target-latency", type=float, help="Target read latency (ns) [Legacy]")
    p.add_argument("--target-energy",  type=float, help="Target write energy (nJ) [Legacy]")
    p.add_argument("--node",           type=int, default=32, choices=[22, 32, 45, 65])
    p.add_argument("--roadmap",        default="HP", choices=["HP", "LOP", "LSTP"])
    p.add_argument("--temperature",    type=float, default=350.0)
    p.add_argument("--steps",          type=int,   default=300)
    p.add_argument("--restarts",       type=int,   default=4)
    p.add_argument("--output",         default=None, help="CSV to append results to")
    p.add_argument("--feasibility",    action="store_true")
    p.add_argument("--verbose-opt",    action="store_true",
                   help="Print Gumbel argmax / post-snap parameter table")
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

    optimizer = InverseOptimizerGumbel(args.tech, use_feasibility=args.feasibility)
    design, ppa, pre_snap, snapped_ppa = optimizer.optimize(
        targets, context,
        steps=args.steps, n_restarts=args.restarts,
        verbose=args.verbose_opt,
    )

    row = {"tech": args.tech, "node_nm": args.node,
           "roadmap": args.roadmap, "temperature_K": args.temperature}
    row.update(design)
    row.update({
        f"pred_{k}": ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_LABELS)
    })
    row.update({
        f"post_snap_pred_{k}": snapped_ppa.get(label) for k, label in zip(TARGET_KEYS, TARGET_LABELS)
    })
    row.update({
        f"target_{k}": targets.get(k, float("nan")) for k in TARGET_KEYS
    })
    row.update({f"pre_snap_{k}": v for k, v in pre_snap.items()})

    df_out = pd.DataFrame([row])
    if args.output:
        df_out.to_csv(args.output, mode="a", index=False,
                      header=not os.path.exists(args.output))
        print(f"Result appended to {args.output}", file=sys.stderr)
    else:
        print(df_out.to_csv(index=False), end="")

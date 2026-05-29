#!/usr/bin/env python3
"""train_model.py — DESTINY surrogate ML model"""

import argparse, json, os, pickle, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

# ── Column definitions ────────────────────────────────────────────────────────

from destiny_utils import TARGET_COLS, TARGET_LABELS

DROP_COLS = [
    # data_num_row_per_set has no DESTINY force flag; always 1 in normal-access-mode caches.
    "data_num_row_per_set",
    "variant_name", "opt_target", "cache_access_mode",
    "cache_miss_latency_ns",
    "cache_miss_energy_nJ",
    "cache_refresh_power_W", "CellInput_MemCellType", "CellInput_ProcessNode",
    "data_total_mats", "data_total_banks", "data_num_row_subarray", "data_num_col_subarray",
    "data_subarray_num_row", "data_subarray_num_col",
    "is_valid",
    # Tag PPA sub-components — directly additive into cache targets (cache_area = data + tag),
    "tag_bank_area_mm2",
    "tag_read_latency_ns", "tag_write_latency_ns",
    "tag_read_energy_pJ", "tag_write_energy_pJ",
    "tag_leakage_mW",
    # Tag subarray dimension outputs — same reason data equivalents are dropped.
    "tag_num_row_subarray", "tag_num_col_subarray",
    "tag_subarray_num_row", "tag_subarray_num_col",
]

CATEGORICAL_COLS = [
    # Technology / process (always one-hot)
    "mem_cell_type", "device_roadmap", "process_node_nm",
    # Cell-level categoricals
    "CellInput_AccessType", "CellInput_ReadMode", "CellInput_ResetMode",
    "CellInput_SetMode", "CellInput_ReadFloating",
    # Data array wire configuration — string-valued outputs from DESTINY
    "data_local_wire_type", "data_local_wire_repeater_type", "data_local_wire_low_swing",
    "data_global_wire_type", "data_global_wire_repeater_type", "data_global_wire_low_swing",
    # Data buffer design style (latency / balanced / area)
    "data_area_optimization_level",
    # Tag array wire configuration (same treatment as data wires)
    "tag_local_wire_type", "tag_local_wire_repeater_type", "tag_local_wire_low_swing",
    "tag_global_wire_type", "tag_global_wire_repeater_type", "tag_global_wire_low_swing",
    # Tag buffer design style
    "tag_area_optimization_level",
]

FORCE_NUMERIC_COLS = ["CellInput_ResetVoltage (V)", "CellInput_SetVoltage (V)", "CellInput_ReadVoltage (V)"]

LOG_NUMERIC_COLS = [
    "capacity_kb", 
    "CellInput_CellArea (F^2)", "CellInput_SRAMCellNMOSWidth (F)", "CellInput_SRAMCellPMOSWidth (F)",
    "CellInput_AccessCMOSWidth (F)", "CellInput_ResistanceOnAtSetVoltage (ohm)",
    "CellInput_ResistanceOffAtSetVoltage (ohm)", "CellInput_ResistanceOnAtResetVoltage (ohm)",
    "CellInput_ResistanceOffAtResetVoltage (ohm)", "CellInput_ResistanceOnAtReadVoltage (ohm)",
    "CellInput_ResistanceOffAtReadVoltage (ohm)", "CellInput_ResistanceOnAtHalfResetVoltage (ohm)",
    "CellInput_CapacitanceOn (F)", "CellInput_CapacitanceOff (F)", "CellInput_DRAMCellCapacitance (F)",
    "CellInput_ReadEnergy (pJ)", "CellInput_ResetEnergy (pJ)", "CellInput_SetEnergy (pJ)",
]

LOG2_CFG_COLS = [
    "word_width_bits", "associativity", "data_stacked_die_count",
    "data_mux_sense_amp", "data_mux_output_lev1", "data_mux_output_lev2",
    "data_num_active_mat_per_row", "data_num_active_mat_per_col",
    # Tag array power-of-2 knobs (mirror of data equivalents)
    "tag_num_row_mat", "tag_num_col_mat",
    "tag_mux_sense_amp", "tag_mux_output_lev1", "tag_mux_output_lev2",
    "tag_num_active_mat_per_row", "tag_num_active_mat_per_col",
]

# Columns that pass through untransformed (linear scale):
# - internal_sensing: binary (0/1), dropped for SRAM/eDRAM via TECH_DROP_COLS
# - temperature_K, data_num_active_*: linear numeric
# - data_mux_output_lev1 is power-of-2 - handled by LOG2_CFG_COLS
# - wire / buffer cols are string categoricals - handled by CATEGORICAL_COLS
LINEAR_NUMERIC_COLS = [
    "temperature_K",
    "data_num_active_subarray_per_row", "data_num_active_subarray_per_col",
    # Tag active-subarray counts (same treatment as data equivalents)
    "tag_num_active_subarray_per_row", "tag_num_active_subarray_per_col",
    "CellInput_ReadVoltage (V)",
]

_NVM_DROPS = [
    "CellInput_ResistanceOnAtSetVoltage (ohm)",  "CellInput_ResistanceOffAtSetVoltage (ohm)",
    "CellInput_ResistanceOnAtResetVoltage (ohm)", "CellInput_ResistanceOffAtResetVoltage (ohm)",
    "CellInput_ResistanceOnAtReadVoltage (ohm)",  "CellInput_ResistanceOffAtReadVoltage (ohm)",
    "CellInput_ResistanceOnAtHalfResetVoltage (ohm)",
    "CellInput_ResetVoltage (V)", "CellInput_SetVoltage (V)", # REMOVING "CellInput_ReadVoltage (V)",
    "CellInput_ReadMode", "CellInput_ResetMode", "CellInput_SetMode",
    "CellInput_ResetPulse (ns)", "CellInput_SetPulse (ns)",
    "CellInput_ResetEnergy (pJ)", "CellInput_SetEnergy (pJ)",
    "CellInput_ReadFloating", "CellInput_VoltageDropAccessDevice (V)",
]
_RRAM_SRAM_DROPS = [
    "CellInput_SRAMCellNMOSWidth (F)", "CellInput_SRAMCellPMOSWidth (F)",
    "CellInput_DRAMCellCapacitance (F)", "CellInput_RetentionTime (us)",
]
_SRAM_EDRAM_EXTRA = ["CellInput_AccessType", "internal_sensing"]

TECH_DROP_COLS = {
    "SRAM":  _NVM_DROPS + ["CellInput_DRAMCellCapacitance (F)", "CellInput_RetentionTime (us)"] + _SRAM_EDRAM_EXTRA,
    "eDRAM": _NVM_DROPS + _SRAM_EDRAM_EXTRA,
    "RRAM":  _RRAM_SRAM_DROPS + ["CellInput_MinSenseVoltage (mV)"],
}


# ── Model ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → skip."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.net(x))


class PPA_MLP(nn.Module):
    """
    Stack of residual blocks with a linear input projection and output head.
    Use forward_with_feasibility() for joint PPA + feasibility inference.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, n_blocks: int = 6,
                 dropout: float = 0.1, has_feasibility_head: bool = False):
        super().__init__()
        self.input_proj           = nn.Linear(input_dim, hidden_dim)
        self.blocks               = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_head          = nn.Linear(hidden_dim, len(TARGET_COLS))
        self.has_feasibility_head = has_feasibility_head
        if has_feasibility_head:
            self.feasibility_head = nn.Linear(hidden_dim, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared backbone: input projection + all residual blocks."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PPA-only forward — backward-compatible with all existing call sites."""
        return self.output_head(self._encode(x))

    def forward_with_feasibility(self, x: torch.Tensor):
        """
        Two-head forward. Returns (ppa_log10 [B,4], p_feasible [B,1]).
        p_feasible is in (0, 1) — probability that the design is physically valid.
        Requires has_feasibility_head=True.
        """
        h = self._encode(x)
        return self.output_head(h), torch.sigmoid(self.feasibility_head(h))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def apply_transforms(df):
    """Apply log10 (physical params) and log2 (power-of-2 knobs) transforms in place."""
    for cols, fn in [
        (LOG_NUMERIC_COLS, lambda x: np.log10(x.clip(lower=1e-12))),
        (LOG2_CFG_COLS,    lambda x: np.log2(x.clip(lower=1))),
    ]:
        present = [c for c in cols if c in df.columns]
        if present:
            df[present] = df[present].apply(pd.to_numeric, errors="coerce").fillna(0).pipe(fn)
    return df


def build_features(df, extra_drop_cols=None):
    """Full feature engineering pipeline: coerce → derived features → drop → transform → one-hot → interactions."""
    df = df.copy()

    # Coerce mixed-type voltage columns to float
    present = [c for c in FORCE_NUMERIC_COLS if c in df.columns]
    if present:
        df[present] = df[present].apply(pd.to_numeric, errors="coerce")

    # Derived features — computed before log transforms
    if "capacity_kb" in df.columns:
        df["derived_sqrt_capacity"] = np.sqrt(df["capacity_kb"])
    if "CellInput_CellArea (F^2)" in df.columns:
        df["derived_sqrt_area"]     = np.sqrt(df["CellInput_CellArea (F^2)"])
    if "CellInput_ReadVoltage (V)" in df.columns:
        df["derived_read_v_sq"]     = df["CellInput_ReadVoltage (V)"] ** 2
    if all(c in df.columns for c in ["capacity_kb", "data_stacked_die_count"]):
        df["derived_cap_per_die"]   = df["capacity_kb"] / df["data_stacked_die_count"]
    if all(c in df.columns for c in ["capacity_kb", "word_width_bits", "data_stacked_die_count"]):
        df["derived_rows_per_die"]  = (df["capacity_kb"] * 1024) / (df["word_width_bits"] * df["data_stacked_die_count"])

    # Drop targets, metadata, structural DESTINY outputs, and tech-specific columns
    drop_list = set(TARGET_COLS + DROP_COLS + (extra_drop_cols or []))
    df = df.drop(columns=[c for c in drop_list if c in df.columns])

    # Log transforms (capacity_kb becomes log10 here)
    df = apply_transforms(df)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], dummy_na=False)

    # Roadmap × log10(capacity) interactions — capacity_kb is already log10 at this point
    for rm_col in [c for c in df.columns if c.startswith("device_roadmap_") and "_x_" not in c]:
        df[f"{rm_col}_x_log10_cap"] = df[rm_col] * df["capacity_kb"]

    return df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)


def log_targets(df):
    """Convert physical target values to log10 space for training."""
    return np.log10(np.clip(df[TARGET_COLS].values.astype(np.float64), 1e-12, None)).astype(np.float32)

def unscale_targets(y_log):
    """Convert log10 model outputs back to physical units."""
    return 10.0 ** y_log

def predict(model, scaler, feat_cols, df_input):
    """End-to-end inference: raw DataFrame → physical-unit predictions."""
    model.eval()
    X = build_features(df_input)
    for col in feat_cols:
        if col not in X.columns: X[col] = 0.0  # fill missing columns with zero
    X_scaled = scaler.transform(X[feat_cols].values).astype(np.float32)
    with torch.no_grad():
        return unscale_targets(model(torch.from_numpy(X_scaled)).cpu().numpy())

# ── Data loading & splitting ──────────────────────────────────────────────────

def filter_physical_failures(df):
    """Remove rows with non-physical simulation outputs (DESTINY solver failures)."""
    n = len(df)
    df = df[
        (df["cache_hit_latency_ns"] < 100) & (df["cache_area_mm2"] < 1000) &
        (df["cache_write_energy_nJ"]  < 1000) & (df["cache_leakage_mW"] > 0) &
        (df["cache_leakage_mW"] < 1e7)
    ]
    if len(df) < n:
        print(f"FILTER: Dropped {n - len(df)} non-physical simulation failures.")
    return df


def prepare_data(args):
    """Load CSV, filter, feature-engineer, and return scaled 80/10/10 train/val/test matrices."""
    print("DESTINY PPA Surrogate Model Training:")
    df = pd.read_csv(args.data)
    if not getattr(args, "feasibility", False):
        df = df.dropna(subset=TARGET_COLS)
        df = filter_physical_failures(df)
    else:
        # Keep invalid designs; failed rows may have NaN mem_cell_type.
        if args.tech != "ALL":
            df["mem_cell_type"] = df["mem_cell_type"].fillna(args.tech)
    print(f"  Final training set size: {df.shape}")

    print("\nTechnology distribution:")
    for tech, count in df["mem_cell_type"].value_counts().items():
        print(f"  {tech:<8}  {count:>5} rows  ({100*count/len(df):.1f}%)")

    # Optionally filter to a single technology and subsample for faster debugging
    if args.tech != "ALL":
        df = df[df["mem_cell_type"] == args.tech]
        print(f"\nFiltered to {args.tech}: {len(df)} rows")
        if len(df) == 0: sys.exit(f"ERROR: No data for '{args.tech}'.")
        if args.sample_size and args.sample_size < len(df):
            df = df.sample(n=args.sample_size, random_state=42)
            print(f"Sub-sampled to {args.sample_size} rows.")

    # Stratified 80/10/10 split — stratify on mem_cell_type to preserve tech balance
    def split(idx, size, strat):
        return train_test_split(idx, test_size=size, random_state=42, stratify=strat)

    idx   = np.arange(len(df))
    strat = df["mem_cell_type"].values
    iv, idx_test       = split(idx, 0.1,   strat)
    idx_train, idx_val = split(iv,  0.111, strat[iv])
    print(f"\nSplit: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test")

    # Build features, dropping tech-irrelevant columns
    extra_drops = (["mem_cell_type"] if args.tech != "ALL" else []) + TECH_DROP_COLS.get(args.tech, [])
    X_df = build_features(df, extra_drop_cols=extra_drops)

    # Remove zero-variance and perfectly correlated columns (uninformative / redundant)
    zero_var = [c for c in X_df.columns if X_df[c].std() == 0]
    if zero_var:
        print(f"Dropping {len(zero_var)} zero-variance columns: {zero_var}")
        X_df.drop(columns=zero_var, inplace=True)

    upper     = X_df.corr().abs().where(np.triu(np.ones((len(X_df.columns),)*2, dtype=bool), k=1))
    redundant = [c for c in upper.columns if any(upper[c] >= 0.99999)]
    if redundant:
        print(f"Dropping {len(redundant)} perfectly correlated columns: {redundant}")
        X_df.drop(columns=redundant, inplace=True)

    feats = list(X_df.columns)
    X_all = X_df.values
    print(f"Feature dimension: {len(feats)}")

    # Fit scaler on train only; apply to val and test
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_train]).astype(np.float32)
    X_val   = scaler.transform(X_all[idx_val]).astype(np.float32)
    X_test  = scaler.transform(X_all[idx_test]).astype(np.float32)

    y_all = log_targets(df)
    if getattr(args, "feasibility", False):
        if "is_valid" not in df.columns:
            sys.exit("ERROR: --feasibility requires an 'is_valid' column in the dataset.")
        y_feas_all = df["is_valid"].values.astype(np.float32)
        y_all = np.concatenate([y_all, y_feas_all[:, None]], axis=1)

    return X_train, X_val, X_test, y_all[idx_train], y_all[idx_val], y_all[idx_test], scaler, feats

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_history(log_rows, save_path, title="PPA Training Loss History", is_tuning=False):
    """Plot per-target loss curves. In tuning mode adds a second set of rows showing Val Log10 MSE."""
    if not log_rows: return
    df = pd.DataFrame(log_rows)
    
    labels     = TARGET_LABELS + ["Total Weighted"]
    train_keys = [f"train_{l}" for l in TARGET_LABELS] + ["train_loss"]
    val_huber  = [f"val_{l}"   for l in TARGET_LABELS] + ["val_loss"]
    val_mse    = [f"val_log_mse_{l}" for l in TARGET_LABELS] + ["val_log_mse_total"]

    n_cols = 5
    n_rows_per_set = (len(labels) + n_cols - 1) // n_cols  # 2 rows for 9 labels
    total_rows = n_rows_per_set * (2 if is_tuning else 1)
    
    fig, axes = plt.subplots(total_rows, n_cols, figsize=(24, 4 * total_rows), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Hide any unused subplots
    for r in range(total_rows):
        for c in range(n_cols):
            set_idx = r // n_rows_per_set
            r_set = r % n_rows_per_set
            elem_idx = r_set * n_cols + c
            if elem_idx >= len(labels):
                axes[r, c].set_visible(False)

    # Plot Huber loss (first set of rows)
    for idx, (label, tk, vk) in enumerate(zip(labels, train_keys, val_huber)):
        r_set = idx // n_cols
        c = idx % n_cols
        ax = axes[r_set, c]
        ax.plot(df["epoch"], df[tk], color="#1f77b4", lw=1.5, label="Train Huber")
        ax.plot(df["epoch"], df[vk], color="#d62728", lw=1.5, label="Val Huber", alpha=0.8)
        ax.set(title=label, yscale="log", xlabel="Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        if c == 0:
            ax.set_ylabel("Huber Loss", fontsize=11)

    # Plot Log10 MSE (second set of rows, tuning only)
    if is_tuning:
        for idx, (label, mk) in enumerate(zip(labels, val_mse)):
            r_set = idx // n_cols
            c = idx % n_cols
            r = n_rows_per_set + r_set
            ax = axes[r, c]
            ax.plot(df["epoch"], df[mk], color="#2ca02c", lw=1.5, label="Val Log10 MSE")
            ax.set(title=label, yscale="log", xlabel="Epoch")
            ax.legend(fontsize=9)
            ax.grid(True, which="both", ls="--", alpha=0.4)
            if c == 0:
                ax.set_ylabel("Log10 MSE", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Evaluation & artifact saving ──────────────────────────────────────────────

def save_and_evaluate(model, X_eval, y_eval, device, args, scaler, feats, log_rows, set_name="val", is_tuning=False):
    """Run inference on eval set, compute metrics, save model artifacts and plots."""
    model.eval()
    if getattr(args, "feasibility", False):
        # Extract only the PPA targets for evaluation
        y_eval_ppa = y_eval[:, :len(TARGET_COLS)]
        with torch.no_grad():
            y_pred_log, y_pred_feas = model.forward_with_feasibility(torch.from_numpy(X_eval).to(device))
            y_pred_log = y_pred_log.cpu().numpy()
            y_pred_feas = y_pred_feas.cpu().numpy()
        y_pred, y_true = unscale_targets(y_pred_log), unscale_targets(y_eval_ppa)
        
        # Evaluate feasibility head accuracy
        feas_acc = ((y_pred_feas > 0.5) == y_eval[:, len(TARGET_COLS):len(TARGET_COLS)+1]).mean()
        print(f"\nFeasibility Classification Accuracy: {feas_acc * 100:.2f}%")
        
        # Save feasibility metrics
        feas_metrics = pd.DataFrame([{"Metric": "Feasibility", "Accuracy_percent": round(feas_acc * 100, 2)}])
        feas_metrics.to_csv(os.path.join(args.output_dir, f"{set_name}_feasibility_metrics.csv"), index=False)
        
        # Plot Feasibility Probability Distribution and Scatter
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            y_true_feas = y_eval[:, len(TARGET_COLS):len(TARGET_COLS)+1].flatten()
            y_pred_feas_flat = y_pred_feas.flatten()
            
            # Panel 1: Histogram
            bins = np.linspace(0, 1, 51)
            ax1.hist(y_pred_feas_flat[y_true_feas == 1], bins=bins, alpha=0.6, label='True Valid', color='green')
            ax1.hist(y_pred_feas_flat[y_true_feas == 0], bins=bins, alpha=0.6, label='True Invalid', color='red')
            ax1.axvline(x=0.5, color='black', linestyle='--')
            ax1.set_title("Probability Distribution")
            ax1.set_xlabel("Predicted Probability")
            ax1.set_ylabel("Count")
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Panel 2: Shape scatter plot
            # Plot a random subset of up to 150 points so it's not a giant unreadable blob
            n_plot = min(len(y_true_feas), 150)
            idx = np.random.choice(len(y_true_feas), n_plot, replace=False)
            
            y_true_sub = y_true_feas[idx]
            y_pred_class = (y_pred_feas_flat[idx] > 0.5).astype(int)
            
            ax2.scatter(range(n_plot), y_true_sub, marker='o', s=80, alpha=0.4, color='blue', label='Ground Truth')
            ax2.scatter(range(n_plot), y_pred_class, marker='x', s=40, color='red', label='Predicted Class')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Invalid (0)', 'Valid (1)'])
            ax2.set_xlabel("Sample Index (Random Subset)")
            ax2.set_title(f"Truth vs Prediction ({n_plot} points)")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{set_name}_feasibility_dist.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot feasibility distribution: {e}")
            
    else:
        with torch.no_grad():
            y_pred_log = model(torch.from_numpy(X_eval).to(device)).cpu().numpy()
        y_pred, y_true = unscale_targets(y_pred_log), unscale_targets(y_eval)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if getattr(args, "feasibility", False):
            valid_mask = (y_eval[:, len(TARGET_COLS)] == 1.0)
            yt = y_true[valid_mask]
            yp = y_pred[valid_mask]
            ye = y_eval[valid_mask]
            ypl = y_pred_log[valid_mask]
        else:
            yt, yp, ye, ypl = y_true, y_pred, y_eval, y_pred_log
            
        metrics_rows = [
            {
                "Metric":            label,
                "R2":                round(r2_score(yt[:,i], yp[:,i]), 4) if len(yt) else 0.0,
                "MAPE_percent":      round(mean_absolute_percentage_error(yt[:,i], yp[:,i]) * 100, 2) if len(yt) else 0.0,
                "MedRelErr_percent": round(np.median(np.abs(yp[:,i]-yt[:,i]) / (yt[:,i]+1e-12)) * 100, 2) if len(yt) else 0.0,
                "MAE":               round(mean_absolute_error(yt[:,i], yp[:,i]), 8) if len(yt) else 0.0,
                "MSE":               round(mean_squared_error(yt[:,i], yp[:,i]), 8) if len(yt) else 0.0,
                "Log10_MSE":         round(mean_squared_error(ye[:,i], ypl[:,i]), 8) if len(yt) else 0.0,
                "True_Min": yt[:,i].min() if len(yt) else 0.0, "True_Max": yt[:,i].max() if len(yt) else 0.0,
                "Pred_Min": yp[:,i].min() if len(yp) else 0.0, "Pred_Max": yp[:,i].max() if len(yp) else 0.0,
            }
            for i, label in enumerate(TARGET_LABELS)
        ]

    os.makedirs(args.output_dir, exist_ok=True)
    plot_training_history(log_rows, os.path.join(args.output_dir, "loss_history.png"),
                          title=f"Training History ({args.tech})", is_tuning=is_tuning)

    metrics_path = os.path.join(args.output_dir, f"{set_name}_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    if log_rows:
        pd.DataFrame(log_rows).to_csv(os.path.join(args.output_dir, "training_log.csv"), index=False)

    # Save model weights, input scaler, and feature column list for inference
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(args.output_dir, "feature_cols.json"), "w") as f: json.dump(feats, f, indent=2)

    print(f"\nDone. Metrics saved to {metrics_path}")

    # Flatten metrics dict for easy consumption by Optuna objective
    final_metrics = {}
    for m in metrics_rows:
        final_metrics |= {f"{m['Metric']}_R2": m["R2"], f"{m['Metric']}_MAPE": m["MAPE_percent"],
                          f"{m['Metric']}_MSE": m["MSE"], f"{m['Metric']}_MAE": m["MAE"],
                          f"{m['Metric']}_Log10_MSE": m["Log10_MSE"]}

    return (model, scaler, feats, final_metrics,
            {l: y_true[:,i] for i,l in enumerate(TARGET_LABELS)},
            {l: y_pred[:,i] for i,l in enumerate(TARGET_LABELS)})

# ── Main training loop ────────────────────────────────────────────────────────

def train(args, trial=None):
    torch.set_num_threads(24)
    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feats = prepare_data(args)

    def make_loader(X, y, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                          batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    device       = torch.device("cpu")
    model        = PPA_MLP(len(feats), args.hidden_dim, args.n_blocks, args.dropout,
                           has_feasibility_head=getattr(args, "feasibility", False)).to(device)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_weights = torch.tensor(args.alpha, dtype=torch.float32, device=device)
    criterion    = nn.HuberLoss(delta=0.5, reduction="none")  # per-element, weighted below
    feas_criterion = nn.BCELoss() if getattr(args, "feasibility", False) else None

    # Cosine annealing with linear warmup — stabilises early training
    warmup    = 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (
        (e+1)/warmup if e < warmup
        else 0.5*(1.0 + np.cos(np.pi*(e-warmup)/max(1, args.epochs-warmup)))
    ))

    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if getattr(args, "feasibility", False):
        print(f"Mode: Two-Head (PPA + Feasibility)")
    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience}) ...")
    print(f"{'Epoch':>6}  {'Train(Huber)':>14}  {'Val Log10MSE':>14}  {'LR':>10}")
    print("-" * 50)

    best_val, patience_count, best_state, log_rows = float("inf"), 0, None, []

    for epoch in range(1, args.epochs + 1):

        # ── Train step: Huber loss with per-target weights ──
        model.train()
        t_loss, t_indiv = 0.0, torch.zeros(len(TARGET_COLS), device=device)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            if getattr(args, "feasibility", False):
                yb_ppa = yb[:, :len(TARGET_COLS)]
                yb_feas = yb[:, len(TARGET_COLS):len(TARGET_COLS)+1]
                
                pred_ppa, pred_feas = model.forward_with_feasibility(xb)
                
                # Feasibility loss
                feas_loss = feas_criterion(pred_feas, yb_feas)
                
                # PPA loss (only on valid designs)
                valid_mask = (yb_feas == 1.0).squeeze()
                if valid_mask.any():
                    indiv = criterion(pred_ppa[valid_mask], yb_ppa[valid_mask]).mean(dim=0)
                    ppa_loss = (indiv * loss_weights).mean()
                else:
                    ppa_loss = torch.tensor(0.0, device=device)
                    indiv = torch.zeros(len(TARGET_COLS), device=device)
                    
                loss = ppa_loss + 10.0 * feas_loss # Give feasibility loss a strong weight
            else:
                indiv = criterion(model(xb), yb).mean(dim=0)   # per-target mean Huber
                loss  = (indiv * loss_weights).mean()           # weighted scalar
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding grads
            optimizer.step()
            t_loss  += loss.item() * len(xb)
            t_indiv += indiv.detach() * len(xb)
        train_loss  = t_loss / len(X_train)
        train_indiv = (t_indiv / len(X_train)).cpu().numpy()

        # ── Validation step: compute both Huber (for plots) and Log10 MSE (for control) ──
        model.eval()
        v_huber, v_huber_i, v_sq = 0.0, torch.zeros(len(TARGET_COLS), device=device), torch.zeros(len(TARGET_COLS), device=device)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                
                if getattr(args, "feasibility", False):
                    yb_ppa = yb[:, :len(TARGET_COLS)]
                    yb_feas = yb[:, len(TARGET_COLS):len(TARGET_COLS)+1]
                    pred_ppa, pred_feas = model.forward_with_feasibility(xb)
                    
                    valid_mask = (yb_feas == 1.0).squeeze()
                    if valid_mask.any():
                        indiv = criterion(pred_ppa[valid_mask], yb_ppa[valid_mask]).mean(dim=0)
                        v_huber   += (indiv * loss_weights).mean().item() * valid_mask.sum().item()
                        v_huber_i += indiv * valid_mask.sum().item()
                        v_sq      += ((pred_ppa[valid_mask] - yb_ppa[valid_mask]) ** 2).sum(dim=0)
                else:
                    pred   = model(xb)
                    indiv  = criterion(pred, yb).mean(dim=0)
                    v_huber   += (indiv * loss_weights).mean().item() * len(xb)
                    v_huber_i += indiv * len(xb)
                    v_sq      += ((pred - yb) ** 2).sum(dim=0)  # accumulate squared errors in log10 space

        # For feasibility, average over valid samples only (approximate here using len(X_val) for simplicity)

        val_huber_loss    = v_huber / len(X_val)
        val_huber_i       = (v_huber_i / len(X_val)).cpu().numpy()
        val_log_mse       = (v_sq / len(X_val)).cpu().numpy()   # per-target Log10 MSE
        val_log_mse_total = float(val_log_mse.sum())             # scalar used for all control decisions

        # Report to Optuna pruner — uses same metric as final trial objective
        if trial is not None:
            trial.report(val_log_mse_total, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Log both Huber and Log10 MSE so plots can show either
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_huber_loss,
               "val_log_mse_total": val_log_mse_total, "lr": current_lr}
        for i, l in enumerate(TARGET_LABELS):
            row[f"train_{l}"]       = train_indiv[i]
            row[f"val_{l}"]         = val_huber_i[i]
            row[f"val_log_mse_{l}"] = val_log_mse[i]
        log_rows.append(row)

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>14.6f}  {val_log_mse_total:>14.6f}  {current_lr:>10.2e}")

        # Early stopping on Log10 MSE — consistent with pruner and Optuna objective
        if val_log_mse_total < best_val - 1e-9:
            best_val, patience_count = val_log_mse_total, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best val Log10 MSE: {best_val:.6f})")
                break

    if best_state:
        model.load_state_dict(best_state)
        print(f"Best weights restored (val Log10 MSE: {best_val:.6f})")

    # Evaluate on test set for final reporting, val set during tuning
    set_name = "test" if args.eval_on_test else "val"
    X_eval   = X_test  if args.eval_on_test else X_val
    y_eval   = y_test  if args.eval_on_test else y_val
    return save_and_evaluate(model, X_eval, y_eval, device, args, scaler, feats, log_rows,
                             set_name=set_name, is_tuning=(trial is not None))

# ── Inference helper ──────────────────────────────────────────────────────────

def load_model(output_dir="model_output"):
    """Load saved model, scaler, and feature list for inference."""
    with open(os.path.join(output_dir, "feature_cols.json")) as f: feat_cols = json.load(f)
    with open(os.path.join(output_dir, "scaler.pkl"), "rb") as f:  scaler   = pickle.load(f)
    model = PPA_MLP(input_dim=len(feat_cols))
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt"), map_location="cpu"))
    return model.eval(), scaler, feat_cols

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(args=None):
    p = argparse.ArgumentParser(description="Train DESTINY PPA surrogate model",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir",   default="model_output")
    p.add_argument("--tech",         default="ALL",        help="SRAM | RRAM | eDRAM | ALL")
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch-size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden-dim",   type=int,   default=256)
    p.add_argument("--n-blocks",     type=int,   default=6)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--patience",     type=int,   default=50)
    p.add_argument("--sample-size",  type=int,   default=0,   help="Subsample N rows (0 = use all).")
    p.add_argument("--log-interval", type=int,   default=20)
    p.add_argument("--alpha",        type=float, nargs=8, default=[1.0]*8,
                                     help="Per-target Huber loss weights: [Area, ReadLat, WriteLat, RefLat, ReadEn, WriteEn, RefEn, Leak].")
    p.add_argument("--from-study",   default=None, help="Load best HPs from named Optuna study.")
    p.add_argument("--eval-on-test", action="store_true", help="Final eval on test set (use once).")
    p.add_argument("--feasibility",  action="store_true", help="Train two-head model on valid+failed data.")
    return p.parse_args(args)


def load_params_from_study(args):
    """Override CLI args with best hyperparameters from a completed Optuna study."""
    if not args.from_study: return
    try:
        best = optuna.load_study(study_name=args.from_study, storage="sqlite:///optuna_study.db").best_params
        print(f"\n[INFO] Loading optimized parameters from: {args.from_study}")
        for k in ["n_blocks", "lr", "weight_decay", "dropout", "hidden_dim"]:
            if k in best: setattr(args, k, best[k])
        best_alphas = []
        for label in TARGET_LABELS:
            clean_label = label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('^', '').replace('/', '_')
            best_alphas.append(best.get(f"alpha_{clean_label}", 1.0))
        if "alpha_lat" in best:
            # Legacy fallback
            best_alphas = [best.get(k, 1.0) for k in ["alpha_lat", "alpha_energy", "alpha_area", "alpha_leak"]] + [1.0] * (len(TARGET_LABELS) - 4)
        args.alpha = best_alphas
    except Exception as e:
        print(f"WARNING: Study '{args.from_study}' load failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    load_params_from_study(args)

    if args.feasibility:
        args.data = f"pareto/{args.tech}/{args.tech}_feasibility.csv"
    else:
        args.data = (f"pareto/{args.tech}/{args.tech}_full_data.csv"
                     if args.tech != "ALL" else "pareto/full_data.csv")

    if args.output_dir == "model_output":
        if args.feasibility:
            args.output_dir = os.path.join("model_output", f"{args.tech.lower()}_feasibility")
        else:
            args.output_dir = os.path.join("model_output", f"{args.tech.lower()}_full_with_data_params")

    if not os.path.exists(args.data):
        sys.exit(f"ERROR: Training data not found: {args.data}")

    train(args)
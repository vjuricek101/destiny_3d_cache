#!/usr/bin/env python3
"""
train_model.py — DESTINY surrogate ML model

Trains MLP to predict (Latency, Area, Energy, Leakage Power) 
Inputs and outputs are both log10-transformed
"""

import argparse
import json
import os
import pickle
import sys
import warnings
import optuna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

# ── Column definitions ────────────────────────────────────────────────────────

TARGET_COLS = [
    "Cache Hit Latency (ns)",
    "Cache Area (mm^2)",
    "Cache Hit Energy (nJ)",
    "Cache Leakage Power (mW)",
]
TARGET_LABELS = ["Latency (ns)", "Area (mm^2)", "Energy (nJ)", "Leakage (mW)"]

DROP_COLS = ["variant_id", "CellInput_MemCellType"]

CATEGORICAL_COLS = [
    "memory_technology",
    "DeviceRoadmap",
    "CellInput_AccessType",
    "CellInput_ReadMode",
    "CellInput_ResetMode",
    "CellInput_SetMode",
    "CellInput_ReadFloating",
]

# Mixed-type columns that may contain strings like "vdd" — coerce to numeric.
FORCE_NUMERIC_COLS = [
    "CellInput_ResetVoltage (V)",
    "CellInput_SetVoltage (V)",
    "CellInput_ReadVoltage (V)",
]

# Wide-range numerics: log10-transformed (zeros stay 0).
LOG_NUMERIC_COLS = [
    "capacity_mb",
    "destiny_total_mats",
    "destiny_total_banks",
    "destiny_mat_rows",
    "destiny_mat_cols",
    "destiny_subarray_rows",
    "destiny_subarray_cols",
    "destiny_bank_stacked",
    "destiny_read_bw_GBs",
    "CellInput_CellArea (F^2)",
    "CellInput_SRAMCellNMOSWidth (F)",
    "CellInput_SRAMCellPMOSWidth (F)",
    "CellInput_AccessCMOSWidth (F)",
    "CellInput_ResistanceOnAtSetVoltage (ohm)",
    "CellInput_ResistanceOffAtSetVoltage (ohm)",
    "CellInput_ResistanceOnAtResetVoltage (ohm)",
    "CellInput_ResistanceOffAtResetVoltage (ohm)",
    "CellInput_ResistanceOnAtReadVoltage (ohm)",
    "CellInput_ResistanceOffAtReadVoltage (ohm)",
    "CellInput_ResistanceOnAtHalfResetVoltage (ohm)",
    "CellInput_CapacitanceOn (F)",
    "CellInput_CapacitanceOff (F)",
    "CellInput_DRAMCellCapacitance (F)",
    "CellInput_ReadEnergy (pJ)",
    "CellInput_ResetEnergy (pJ)",
    "CellInput_SetEnergy (pJ)",
]

# Swept .cfg architectural parameters — log2-transformed (power-of-2 knobs).
# word_width and associativity come from SRAM/eDRAM cache-mode sweeps;
# stacked_die_count from all technologies.
# internal_sensing is a binary flag from RRAM sweeps (0 or 1, kept as-is).
LOG2_CFG_COLS = [
    "word_width",
    "associativity",
    "stacked_die_count",
]
# Binary cfg param — just passthrough (already 0/1).
BINARY_CFG_COLS = ["internal_sensing"]

# Other numeric inputs that scale linearly (e.g., ProcessNode, Temperature)
LINEAR_NUMERIC_COLS = [
    "Temperature (K)",
    "CellInput_ProcessNode",
    "destiny_senseamp_mux",     # small integers, linear is fine
    "destiny_output_mux_l2",
    "destiny_row_activation_num",
    "destiny_col_activation_num",
]


# Columns irrelevant to each specific technology (beyond the shared DROP_COLS).
_RRAM_SRAM_SHARED_DROPS = [
    "CellInput_SRAMCellNMOSWidth (F)", "CellInput_SRAMCellPMOSWidth (F)",
    "CellInput_DRAMCellCapacitance (F)", "CellInput_RetentionTime (us)",
]
_NVM_SHARED_DROPS = [  # columns absent from both SRAM and eDRAM
    "CellInput_ResistanceOnAtSetVoltage (ohm)",  "CellInput_ResistanceOffAtSetVoltage (ohm)",
    "CellInput_ResistanceOnAtResetVoltage (ohm)", "CellInput_ResistanceOffAtResetVoltage (ohm)",
    "CellInput_ResistanceOnAtReadVoltage (ohm)",  "CellInput_ResistanceOffAtReadVoltage (ohm)",
    "CellInput_ResistanceOnAtHalfResetVoltage (ohm)",
    "CellInput_ResetVoltage (V)", "CellInput_SetVoltage (V)", "CellInput_ReadVoltage (V)",
    "CellInput_ReadMode", "CellInput_ResetMode", "CellInput_SetMode",
    "CellInput_ResetPulse (ns)", "CellInput_SetPulse (ns)",
    "CellInput_ResetEnergy (pJ)", "CellInput_SetEnergy (pJ)",
    "CellInput_ReadFloating", "CellInput_VoltageDropAccessDevice (V)",
]

TECH_DROP_COLS: dict[str, list[str]] = {
    "SRAM":  _NVM_SHARED_DROPS + [
        "CellInput_DRAMCellCapacitance (F)", "CellInput_RetentionTime (us)",
        "CellInput_AccessType",
    ],
    "eDRAM": _NVM_SHARED_DROPS + ["CellInput_AccessType"],
    "RRAM":  _RRAM_SRAM_SHARED_DROPS + ["CellInput_MinSenseVoltage (mV)"],
}


# ── Model ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.net(x))


class PPA_MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, n_blocks: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_head = nn.Linear(hidden_dim, len(TARGET_COLS))

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, extra_drop_cols: list = None) -> pd.DataFrame:
    df = df.copy()

    force_cols = [c for c in FORCE_NUMERIC_COLS if c in df.columns]
    if force_cols:
        df[force_cols] = df[force_cols].apply(pd.to_numeric, errors="coerce")

    # Derived features
    derived = {
        "derived_sqrt_capacity": lambda d: np.sqrt(d["capacity_mb"]) if "capacity_mb" in d else None,
        "derived_sqrt_area":     lambda d: np.sqrt(d["CellInput_CellArea (F^2)"]) if "CellInput_CellArea (F^2)" in d else None,
        "derived_read_v_sq":     lambda d: d["CellInput_ReadVoltage (V)"]**2 if "CellInput_ReadVoltage (V)" in d else None,
    }
    for name, func in derived.items():
        res = func(df)
        if res is not None: df[name] = pd.to_numeric(res, errors="coerce").fillna(0)

    drop_list = set(TARGET_COLS + DROP_COLS + (extra_drop_cols or []))
    df = df[[c for c in df.columns if c not in drop_list]]

    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_present, dummy_na=False)

    log_cols = [c for c in LOG_NUMERIC_COLS if c in df.columns]
    if log_cols:
        df[log_cols] = df[log_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df[log_cols] = np.where(df[log_cols] > 0, np.log10(df[log_cols].clip(lower=1e-12)), 0.0)

    # Log2-transform power-of-2 cfg knobs (word_width, associativity, stacked_die_count)
    log2_cols = [c for c in LOG2_CFG_COLS if c in df.columns]
    if log2_cols:
        df[log2_cols] = df[log2_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df[log2_cols] = np.where(df[log2_cols] > 0, np.log2(df[log2_cols].clip(lower=1)), 0.0)

    # Binary cfg params — coerce to numeric (already 0/1)
    bin_cols = [c for c in BINARY_CFG_COLS if c in df.columns]
    if bin_cols:
        df[bin_cols] = df[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Categorical encoding (One-Hot) for roadmap and node
    cat_cols = [c for c in ["roadmap", "ProcessNode"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

    # Final cast to numeric, dropping any non-numeric leftovers
    return df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)


def log_targets(df: pd.DataFrame) -> np.ndarray:
    return np.log10(np.clip(df[TARGET_COLS].values.astype(np.float64), 1e-12, None)).astype(np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def prepare_data(args):
    """Loads, filters, engineers, scales, and splits the data into 80/10/10 matrices."""
    print("DESTINY PPA Surrogate Model Training:")
    df_raw = pd.read_csv(args.data).dropna(subset=TARGET_COLS)
    
    # Physical Filter: Drop DESTINY simulation failures (e.g. Latency > 100ns or overflow)
    initial_len = len(df_raw)
    df_raw = df_raw[
        (df_raw["Cache Hit Latency (ns)"] < 100) & 
        (df_raw["Cache Area (mm^2)"] < 1000) & 
        (df_raw["Cache Hit Energy (nJ)"] < 1000)
    ]
    if len(df_raw) < initial_len:
        print(f"  [FILTER] Dropped {initial_len - len(df_raw)} non-physical simulation failures (Latency > 100ns, etc.)")
    
    print(f"  Final training set size: {df_raw.shape}")

    print("\nTechnology distribution:")
    for tech, count in df_raw["memory_technology"].value_counts().items():
        print(f"  {tech:<8}  {count:>5} rows  ({100*count/len(df_raw):.1f}%)")

    if args.tech != "ALL":
        df_raw = df_raw[df_raw["memory_technology"] == args.tech]
        print(f"\nFiltered to {args.tech}: {len(df_raw)} rows")
        if len(df_raw) == 0:
            sys.exit(f"ERROR: No data found for technology '{args.tech}'.")
        if args.sample_size and args.sample_size < len(df_raw):
            df_raw = df_raw.sample(n=args.sample_size, random_state=42)
            print(f"Sub-sampled to {args.sample_size} rows.")

    # Stratified 80/10/10 split
    def split(indices, size, strat):
        return train_test_split(indices, test_size=size, random_state=42, stratify=strat)

    idx = np.arange(len(df_raw))
    strat = df_raw["memory_technology"].values
    iv, idx_test = split(idx, 0.1, strat)
    idx_train, idx_val = split(iv, 0.111, strat[iv])
    print(f"\nSplit: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test")

    # Feature engineering
    extra_drops = (["memory_technology"] if args.tech != "ALL" else []) + TECH_DROP_COLS.get(args.tech, [])
    X_all_df = build_features(df_raw, extra_drop_cols=extra_drops)

    # Drop perfectly correlated columns
    upper = X_all_df.corr().abs().where(np.triu(np.ones((len(X_all_df.columns),) * 2, dtype=bool), k=1))
    redundant = [c for c in upper.columns if any(upper[c] >= 0.99999)]
    if redundant:
        print(f"Dropping {len(redundant)} perfectly correlated columns: {redundant}")
        X_all_df.drop(columns=redundant, inplace=True)

    feature_names = list(X_all_df.columns)
    X_all = X_all_df.values
    print(f"Feature dimension: {len(feature_names)}")

    # Scale Inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_train]).astype(np.float32)
    X_val   = scaler.transform(X_all[idx_val]).astype(np.float32)
    X_test  = scaler.transform(X_all[idx_test]).astype(np.float32)

    # Scale Targets
    y_all = log_targets(df_raw)
    y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


def plot_validation(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, title: str = "PPA Validation Scatter"):
    """Generates a 2x2 residual scatter plot (% error vs. true value) for the 4 PPA metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for ax, i in zip(axes.flat, range(len(TARGET_LABELS))):
        error_pct = (y_pred[:, i] - y_true[:, i]) / (y_true[:, i] + 1e-12) * 100
        median_err = np.median(error_pct)

        ax.axhline(0,          color="red",    linestyle="--", lw=1.5, label="Ideal (0% error)")
        ax.axhline(median_err, color="orange", linestyle="--", lw=1.5, label=f"Median ({median_err:+.1f}%)")
        ax.axhspan(-10, 10,    color="green",  alpha=0.08,             label="±10% band")

        ax.scatter(y_true[:, i], error_pct, alpha=0.5, edgecolors="w", s=30)

        r2   = r2_score(y_true[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]) * 100
        ax.text(0.03, 0.97, f"R²={r2:.3f}  MAPE={mape:.1f}%",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_title(TARGET_LABELS[i])
        ax.set_xlabel(f"True {TARGET_LABELS[i]}")
        ax.set_ylabel("Error (%)")
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_and_evaluate(model, X_test, y_test, device, args, scaler, feature_names, log_rows):
    """Evaluates testing holdouts, computes unscaled metrics, and saves model artifacts."""
    model.eval()
    with torch.no_grad():
        y_pred = 10.0 ** model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    y_true = 10.0 ** y_test

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics_rows = [
            {
                "Metric": label,
                "R2":                round(r2_score(y_true[:, i], y_pred[:, i]), 4),
                "MAPE_percent":      round(mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]) * 100, 2),
                "MedRelErr_percent": round(np.median(np.abs(y_pred[:, i] - y_true[:, i]) / (y_true[:, i] + 1e-12)) * 100, 2),
                "MAE":               round(mean_absolute_error(y_true[:, i], y_pred[:, i]), 8),
                "MSE":               round(mean_squared_error(y_true[:, i], y_pred[:, i]), 8),
                "True_Min": y_true[:, i].min(), "True_Max": y_true[:, i].max(),
                "Pred_Min": y_pred[:, i].min(), "Pred_Max": y_pred[:, i].max(),
            }
            for i, label in enumerate(TARGET_LABELS)
        ]

    # Save artifacts & Plots
    plot_validation(y_true, y_pred, os.path.join(args.output_dir, "validation_scatter.png"), title=f"Testing Holdout Correlation ({args.tech})")
    
    metrics_path = os.path.join(args.output_dir, "test_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    
    if log_rows:
        pd.DataFrame(log_rows).to_csv(os.path.join(args.output_dir, "training_log.csv"), index=False)
        
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.output_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\nDone. Metrics saved to {metrics_path}")

    final_metrics = {}
    for m in metrics_rows:
        final_metrics |= {f"{m['Metric']}_R2": m["R2"], f"{m['Metric']}_MAPE": m["MAPE_percent"],
                          f"{m['Metric']}_MSE": m["MSE"], f"{m['Metric']}_MAE": m["MAE"]}

    y_true_dict = {label: y_true[:, i] for i, label in enumerate(TARGET_LABELS)}
    y_pred_dict = {label: y_pred[:, i] for i, label in enumerate(TARGET_LABELS)}
    return model, scaler, feature_names, final_metrics, y_true_dict, y_pred_dict


def train(args, trial=None):
    torch.set_num_threads(24)
    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feats = prepare_data(args)

    def make_loader(X, y, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                          batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    # Model Data
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = PPA_MLP(len(feats), args.hidden_dim, args.n_blocks, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    warmup = 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (
        (e + 1) / warmup if e < warmup
        else 0.5 * (1.0 + np.cos(np.pi * (e - warmup) / (args.epochs - warmup)))
    ))

    loss_weights = torch.tensor(args.alpha, dtype=torch.float32, device=device)
    base_criterion = nn.HuberLoss(delta=0.5, reduction="none")

    def criterion(pred, true):
        return (base_criterion(pred, true) * loss_weights).mean()

    print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience}) ...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print("-" * 46)

    best_val_loss, patience_count, best_state, log_rows = float("inf"), 0, None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(xb)
        train_loss = total_train_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)
                           for xb, yb in val_loader) / len(X_val)

        if trial is not None:
            import optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>10.2e}")

        if val_loss < best_val_loss - 1e-7:
            best_val_loss, patience_count = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
                break

    if best_state:
        model.load_state_dict(best_state)
        print(f"Best weights restored (val loss: {best_val_loss:.6f})")

    return save_and_evaluate(model, X_test, y_test, device, args, scaler, feats, log_rows)


# ── Inference helper ──────────────────────────────────────────────────────────

def load_model(output_dir: str = "model_output"):
    """Load a saved model for inference. Returns (model, scaler, feat_cols)."""
    with open(os.path.join(output_dir, "feature_cols.json")) as f:
        feat_cols = json.load(f)
    with open(os.path.join(output_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    model = PPA_MLP(input_dim=len(feat_cols))
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt"), map_location="cpu"))
    return model.eval(), scaler, feat_cols


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(args=None):
    p = argparse.ArgumentParser(
        description="Train DESTINY PPA surrogate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",   default="model_output", help="Directory for saved artifacts.")
    p.add_argument("--tech",         default="ALL",          help="Filter to one memory technology (SRAM, RRAM, eDRAM).")
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch-size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden-dim",   type=int,   default=512)
    p.add_argument("--n-blocks",     type=int,   default=6)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--patience",     type=int,   default=50)
    p.add_argument("--sample-size",  type=int,   default=0)
    p.add_argument("--log-interval", type=int,   default=20)
    p.add_argument("--alpha", type=float, nargs=4, default=[1.5, 1.0, 3.0, 1.0],
                   help="Loss weights for [Latency, Area, Energy, Leakage].")
    p.add_argument("--from-study", default=None, help="Load optimized hyperparameters from an Optuna study.")
    p.add_argument("--arch", action="store_true", help="Use architectural sweep data source.")
    return p.parse_args(args)


def load_params_from_study(args):
    if not args.from_study: return
    db = "sqlite:///optuna_study.db"
    try:
        import optuna
        best = optuna.load_study(study_name=args.from_study, storage=db).best_params
        print(f"\n[INFO] Loading optimized parameters from: {args.from_study}")
        for k in ["hidden_dim", "n_blocks", "lr", "dropout"]:
            if k in best: setattr(args, k, best[k])
        if "alpha_lat" in best:
            args.alpha = [best.get(f"alpha_{m}", 1.0) for m in ["lat", "area", "energy", "leak"]]
    except Exception as e:
        print(f"  [WARNING] Study '{args.from_study}' load failed: {e}")

if __name__ == "__main__":
    args = parse_args()
    load_params_from_study(args)

    suffix = "_arch" if args.arch else ""
    args.data = (f"pareto/{args.tech}{suffix}/{args.tech}{suffix}_full_data.csv"
                 if args.tech != "ALL" else "pareto/full_data.csv")

    if args.output_dir == "model_output":
        suffix = "_full" if args.data and "full" in args.data else ""
        args.output_dir = os.path.join("model_output", f"{args.tech.lower()}{suffix}")

    if not os.path.exists(args.data):
        sys.exit(f"ERROR: Training data not found: {args.data}")

    train(args)
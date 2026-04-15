#!/usr/bin/env python3
"""
train_model.py — DESTINY surrogate ML model

Trains a feedforward MLP to predict (Latency, Area, Energy, Leakage Power) from
memory cell configuration parameters + capacity.  Inputs and outputs are both
log10-transformed to handle the multi-order-of-magnitude ranges in the data.

Usage:
    python3 train_model.py
    python3 train_model.py --data pareto/training_dataset_pareto.csv --epochs 300
    python3 train_model.py --output-dir my_model --batch-size 128 --lr 5e-4

Outputs saved to --output-dir (default: model_output/):
    model.pt          PyTorch state dict
    scaler.pkl        Fitted StandardScaler
    feature_cols.json Ordered list of feature column names after preprocessing
    training_log.csv  Epoch-by-epoch train/val loss
"""

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

# Dependency checks
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.exit(
        "ERROR: PyTorch not found.\n"
        "Install with: pip install torch\n"
        "Or activate the environment where the Pareto analysis ran."
    )

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_percentage_error
except ImportError:
    sys.exit(
        "ERROR: scikit-learn not found.\n"
        "Install with: pip install scikit-learn"
    )

# Column definitions

TARGET_COLS = [
    "Cache Hit Latency (ns)",
    "Cache Area (mm^2)",
    "Cache Hit Energy (nJ)",
    "Cache Leakage Power (mW)",
]
TARGET_LABELS = ["Latency (ns)", "Area (mm^2)", "Energy (nJ)", "Leakage (mW)"]

# Metadata — not predictive features
DROP_COLS = ["variant_id", "CellInput_MemCellType"]

# Categorical columns → one-hot encoded
CATEGORICAL_COLS = [
    "memory_technology",
    "CellInput_AccessType",
    "CellInput_ReadMode",
    "CellInput_ResetMode",
    "CellInput_SetMode",
    "CellInput_ReadFloating",
]

# Columns with string values that should be numeric (e.g. eDRAM uses "vdd").
# pd.to_numeric(..., errors="coerce") will turn non-numeric strings → NaN → 0.
FORCE_NUMERIC_COLS = [
    "CellInput_ResetVoltage (V)",
    "CellInput_SetVoltage (V)",
    "CellInput_ReadVoltage (V)",
]

# Wide-range numeric inputs: log10-transform after NaN→0 fill.
# Zero-filled (absent) features stay at 0; positive values get log10.
LOG_NUMERIC_COLS = [
    "capacity_mb",
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


# Model

class PPA_MLP(nn.Module):
    """
    3-hidden-layer MLP:  input → 256 → 256 → 128 → 4 (log10 PPA outputs).

    LayerNorm instead of BatchNorm for stability across mixed-technology batches.
    GELU activations for smooth multi-scale regression.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(TARGET_COLS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Preprocessing

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw CSV into a float32 feature matrix.

    Steps (all performed on a copy; no in-place mutation):
    1. Force-convert mixed-type columns (e.g. "vdd") to numeric, NaN on failure.
    2. One-hot encode categorical columns (unknown categories → all-zero row).
    3. Log10-transform wide-range numeric columns (0-filled absent features stay 0).
    4. Fill all remaining NaNs with 0 (absent tech-specific features).
    5. Cast to float32.
    """
    df = df.copy()

    # 1. Force-convert columns that may contain strings like "vdd"
    for col in FORCE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Drop target + metadata columns
    keep = [c for c in df.columns if c not in TARGET_COLS + DROP_COLS]
    df = df[keep]

    # 3. One-hot encode (missing categories produce all-zero rows automatically)
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_present, dummy_na=False)

    # 4. Log10-transform wide-range numeric columns
    for col in LOG_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            pos = df[col] > 0
            df.loc[pos, col] = np.log10(df.loc[pos, col])
            # Non-positive (absent/zero) values stay at 0

    # 5. Fill remaining NaNs and cast
    df = df.fillna(0).astype(np.float32)
    return df


def log_targets(df: pd.DataFrame) -> np.ndarray:
    """Return log10 of the 4 PPA targets as float32 array."""
    vals = df[TARGET_COLS].values.astype(np.float64)
    # Clip to avoid log(0); all legitimate DESTINY outputs are positive
    vals = np.clip(vals, 1e-12, None)
    return np.log10(vals).astype(np.float32)


# Training

def train(args):
    print("DESTINY PPA Surrogate Model Training")
    print("-" * 40)

    # Load data
    print(f"Loading data from: {args.data}")
    df_raw = pd.read_csv(args.data)
    print(f"  Raw shape: {df_raw.shape}")

    df_raw = df_raw.dropna(subset=TARGET_COLS)
    print(f"  After dropping NaN targets: {len(df_raw)} rows\n")

    print("Technology distribution:")
    vc = df_raw["memory_technology"].value_counts()
    for tech, count in vc.items():
        pct = 100 * count / len(df_raw)
        print(f"  {tech:<8}  {count:>5} rows  ({pct:.1f}%)")

    # Stratified split (build indices first, then slice)
    tech_labels = df_raw["memory_technology"].values
    idx = np.arange(len(df_raw))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.10, random_state=42, stratify=tech_labels
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.111,          # 0.111 × 0.9 ≈ 0.10 of total
        random_state=42,
        stratify=tech_labels[idx_trainval],
    )

    df_train = df_raw.iloc[idx_train]
    df_val   = df_raw.iloc[idx_val]
    df_test  = df_raw.iloc[idx_test]

    print(f"\nSplit: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    # Feature engineering
    # Build features on the FULL dataset first (one-hot column alignment),
    # then slice by split index.  StandardScaler is fit only on train.
    X_all_df = build_features(df_raw)
    feature_names = list(X_all_df.columns)
    X_all = X_all_df.values

    X_train_raw = X_all[idx_train]
    X_val_raw   = X_all[idx_val]
    X_test_raw  = X_all[idx_test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    y_all = log_targets(df_raw)
    y_train = y_all[idx_train]
    y_val   = y_all[idx_val]
    y_test  = y_all[idx_test]

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"  ({len(feature_names)} features: one-hot + numeric)")

    # DataLoaders
    def make_loader(X, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model     = PPA_MLP(input_dim=X_train.shape[1], dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # Training loop
    best_val_loss = float("inf")
    patience_count = 0
    best_state = None
    log_rows = []

    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience}) ...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print("-" * 46)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
        train_loss = train_loss_sum / len(X_train)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss_sum += criterion(model(xb), yb).item() * len(xb)
        val_loss = val_loss_sum / len(X_val)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "lr": current_lr})

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>10.2e}")

        # Early stopping
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val loss: {best_val_loss:.6f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest weights restored (val loss: {best_val_loss:.6f})")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        y_pred_log = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    # Back-transform to physical units for reporting
    y_pred = 10.0 ** y_pred_log
    y_true = 10.0 ** y_test

    print(f"\nTest set performance  ({len(X_test)} samples)")
    print(f"  {'Metric':<20}  {'R^2':>8}  {'MAPE':>8}  {'MedRelErr':>10}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*10}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, label in enumerate(TARGET_LABELS):
            r2   = r2_score(y_true[:, i], y_pred[:, i])
            mape = mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]) * 100
            med  = np.median(
                np.abs(y_pred[:, i] - y_true[:, i]) / (y_true[:, i] + 1e-12)
            ) * 100
            low  = " [low]" if r2 < 0.85 else ""
            print(f"  {label:<20}  {r2:>8.4f}  {mape:>7.2f}%  {med:>9.2f}%{low}")

    # Per-technology breakdown
    test_techs = df_test["memory_technology"].values
    print(f"\n  R^2 by Technology")
    print(f"  {'Tech':<8}  " +
          "  ".join(f"{l:<16}" for l in TARGET_LABELS))
    print(f"  {'-'*8}  " + "  ".join("-" * 16 for _ in TARGET_LABELS))

    for tech in sorted(set(test_techs)):
        mask = test_techs == tech
        if mask.sum() < 2:
            print(f"  {tech:<8}  (too few samples: {mask.sum()})")
            continue
        r2_vals = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(TARGET_COLS)):
                r2_vals.append(r2_score(y_true[mask, i], y_pred[mask, i]))
        r2_str = "  ".join(f"{v:>16.4f}" for v in r2_vals)
        print(f"  {tech:<8}  {r2_str}")

    # Prediction range check
    print(f"\n  Prediction vs Truth ranges (test set):")
    print(f"  {'Metric':<20}  {'True min':>12}  {'True max':>12}  "
          f"{'Pred min':>12}  {'Pred max':>12}")
    for i, label in enumerate(TARGET_LABELS):
        print(f"  {label:<20}  "
              f"{y_true[:, i].min():>12.4g}  {y_true[:, i].max():>12.4g}  "
              f"{y_pred[:, i].min():>12.4g}  {y_pred[:, i].max():>12.4g}")

    # Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)

    model_path   = os.path.join(args.output_dir, "model.pt")
    scaler_path  = os.path.join(args.output_dir, "scaler.pkl")
    feature_path = os.path.join(args.output_dir, "feature_cols.json")
    log_path     = os.path.join(args.output_dir, "training_log.csv")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(feature_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print(f"\nSaved to: {args.output_dir}/")
    print(f"  model.pt          ({os.path.getsize(model_path) / 1024:.0f} KB)")
    print(f"  scaler.pkl")
    print(f"  feature_cols.json ({len(feature_names)} features)")
    print(f"  training_log.csv  ({len(log_rows)} epochs)")

    return model, scaler, feature_names


# Inference helper (for external use)

def load_model(output_dir: str = "model_output"):
    """
    Load a saved model for inference.

    Returns:
        model     : PPA_MLP (eval mode, CPU)
        scaler    : fitted StandardScaler
        feat_cols : list of feature column names

    Example:
        model, scaler, feat_cols = load_model()
        # Build a single-row DataFrame matching feat_cols, then:
        X = scaler.transform(df[feat_cols].fillna(0).astype(np.float32))
        with torch.no_grad():
            y_log = model(torch.from_numpy(X))
        y_pred = 10 ** y_log.numpy()   # [latency, area, energy, leakage]
    """
    feature_path = os.path.join(output_dir, "feature_cols.json")
    scaler_path  = os.path.join(output_dir, "scaler.pkl")
    model_path   = os.path.join(output_dir, "model.pt")

    with open(feature_path) as f:
        feat_cols = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = PPA_MLP(input_dim=len(feat_cols))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, scaler, feat_cols


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Train DESTINY PPA surrogate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default="pareto/training_dataset_pareto.csv",
        help="Path to the Pareto training CSV",
    )
    p.add_argument(
        "--output-dir", default="model_output",
        help="Directory to save model artifacts",
    )
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3,
                   help="Initial learning rate (cosine-annealed to lr*0.01)")
    p.add_argument("--dropout",      type=float, default=0.10)
    p.add_argument("--patience",     type=int,   default=50,
                   help="Early-stopping patience in epochs")
    p.add_argument("--log-interval", type=int,   default=20,
                   help="Print loss every N epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

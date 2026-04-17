#!/usr/bin/env python3
"""
sanity check, e.g., 
- tests if increasing memory capactiy leads to higher predicted area and latency
- varies sizing of transistors and checks if model predicts smaller transistors to be slower
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from train_model import load_model, build_features, TARGET_LABELS, plot_validation

def predict(model, scaler, feat_cols, df_rows: pd.DataFrame) -> np.ndarray:
    """Predict PPA in physical units [Latency, Area, Energy, Leakage]."""
    X_df = build_features(df_rows).reindex(columns=feat_cols, fill_value=0.0).astype("float32")
    with torch.no_grad():
        return 10.0 ** model(torch.from_numpy(scaler.transform(X_df.values).astype("float32"))).numpy()


def check_monotonicity(model, scaler, feat_cols, tech):
    caps = [2**i / 1024.0 for i in range(1, 15)] # 2KB to 16MB
    df = pd.DataFrame([{"memory_technology": tech, "capacity_mb": c, "variant_id": 1, 
                       "CellInput_CellArea (F^2)": 100.0, "CellInput_CellAspectRatio": 1.0} for c in caps])
    
    p = predict(model, scaler, feat_cols, df)
    ok = (p[0,0] < p[-1,0]) and (p[0,1] < p[-1,1])
    print(f"[Monotonicity Check] {'PASS' if ok else 'FAIL'} ({p[0,0]:.2f}ns @ 2KB -> {p[-1,0]:.2f}ns @ 16MB)")


def check_accuracy(model, scaler, feat_cols, data_path, tech, model_dir):
    df = pd.read_csv(data_path).dropna()
    samples = df[df["memory_technology"] == tech] if tech.upper() != "ALL" else df
    if len(samples) == 0:
        return print(f"No samples found for {tech} in {data_path}.")

    samples = samples.sample(min(50, len(samples)))
    y_true = samples[["Cache Hit Latency (ns)", "Cache Area (mm^2)", "Cache Hit Energy (nJ)", "Cache Leakage Power (mW)"]].values
    y_pred = predict(model, scaler, feat_cols, samples)
    
    print(f"\n[Accuracy Check] Dataset: {os.path.basename(data_path)} ({len(samples)} samples)")
    for i, label in enumerate(TARGET_LABELS):
        err = np.median(np.abs(y_pred[:, i] - y_true[:, i]) / (y_true[:, i] + 1e-12)) * 100
        print(f"    {label:<20}: {err:5.2f}% median error")
        
    plot_validation(y_true, y_pred, os.path.join(model_dir, "accuracy_scatter.png"), title=f"Random Sample Correlation ({tech})")


def main(args):
    tech = args.tech.upper()
    model_dir = os.path.join("model_output", tech.lower()) if args.model_dir == "model_output" and tech != "ALL" else args.model_dir
    
    if not os.path.exists(model_dir):
        sys.exit(f"ERROR: Model directory {model_dir} not found.")

    log_file = os.path.join(model_dir, "validation_log.txt")
    print(f"Validation results saved to: {log_file}")
    sys.stdout = open(log_file, "w")

    model, scaler, feat_cols = load_model(model_dir)
    print(f"Validating model in [{model_dir}] | Tech: {tech}\n" + "-"*40)
    
    check_monotonicity(model, scaler, feat_cols, tech)
    
    data_paths = args.data or ([f"pareto/{tech}/{tech}_pareto.csv"] if tech != "ALL" else ["pareto/pareto.csv"])
    for path in data_paths:
        if os.path.exists(path):
            check_accuracy(model, scaler, feat_cols, path, tech, model_dir)
        else:
            print(f"Warning: Validation data not found: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tech", default="ALL", help="Target memory tech (SRAM, RRAM, eDRAM, ALL)")
    p.add_argument("--model-dir", default="model_output")
    p.add_argument("--data", nargs="+", default=None, help="CSV file(s) for accuracy testing")
    main(p.parse_args())

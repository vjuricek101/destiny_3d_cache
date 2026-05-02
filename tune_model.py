#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import optuna
import train_model
import numpy as np

def objective(trial, config):
    """Optuna objective function for MLP hyperparameter search."""
    # Start with default training args
    args = train_model.parse_args(args=[])
    
    # Suggest architecture
    args.tech       = config["tech"]
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    args.n_blocks   = trial.suggest_int("n_blocks", 4, 16)
    args.lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    args.dropout    = trial.suggest_float("dropout", 0.05, 0.2)
    
    # Suggest loss weights for [Latency, Area, Energy, Leakage]
    args.alpha      = [trial.suggest_float("alpha_lat", 1.0, 10.0),
                       trial.suggest_float("alpha_area", 1.0, 10.0),
                       trial.suggest_float("alpha_energy", 1.0, 10.0),
                       trial.suggest_float("alpha_leak", 1.0, 10.0)]
    
    args.data        = config["data"]
    args.sample_size = config["sample_size"]
    args.epochs      = config.get("epochs", 300)
    args.patience    = config.get("patience", 20)
    args.output_dir  = os.path.join("model_output", "tuning", f"{config['study_name']}_trial_{trial.number}")

    # Execution
    _, _, _, metrics, y_true, y_pred = train_model.train(args, trial=trial)  

    # Log10-MSE Calculation
    def log10_mse(true, pred):
        return np.mean((np.log10(np.clip(true, 1e-12, None)) - np.log10(np.clip(pred, 1e-12, None)))**2)

    total_error = sum([
        log10_mse(y_true["Read Latency (ns)"], y_pred["Read Latency (ns)"]),
        log10_mse(y_true["Area (mm^2)"],       y_pred["Area (mm^2)"]),
        log10_mse(y_true["Write Energy (nJ)"], y_pred["Write Energy (nJ)"]),
        log10_mse(y_true["Leakage (mW)"],      y_pred["Leakage (mW)"])
    ])

    return total_error

def main():
    parser = argparse.ArgumentParser(description="Unified Hyperparameter Tuner")
    parser.add_argument("--tech",        default="SRAM", help="Memory technology (SRAM, RRAM, eDRAM, ALL)")
    parser.add_argument("--arch",        action="store_true", help="Tune on architectural sweep data")
    parser.add_argument("--trials",      type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--sample-size", type=int, default=50000, help="Max samples for tuning")
    parser.add_argument("--epochs",      type=int, default=300)
    parser.add_argument("--study-name",  default="destiny_tune_v1")
    args = parser.parse_args()

    # Determine Data Path
    suffix = "_arch" if args.arch else ""
    if args.tech.upper() == "ALL":
        data_path = f"pareto/pareto{suffix}.csv"
    else:
        data_path = f"pareto/{args.tech}{suffix}/{args.tech}{suffix}_full_data.csv"

    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found: {data_path}")
        sys.exit(1)

    config = {
        "data":        data_path,
        "tech":        args.tech.upper(),
        "sample_size": args.sample_size,
        "epochs":      args.epochs,
        "study_name":  args.study_name,
        "is_arch":     args.arch
    }

    study = optuna.create_study(
        study_name=args.study_name, 
        storage="sqlite:///optuna_study.db",
        load_if_exists=True, 
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    print(f"TUNING: Study '{args.study_name}' (Arch Mode: {args.arch})")
    print(f"DATA:   {data_path}")
    
    study.optimize(lambda t: objective(t, config), n_trials=args.trials)
    
    print(f"\nBest Loss: {study.best_value:.6f} | Trial: {study.best_trial.number}")
    for k, v in study.best_params.items():
        print(f"  {k:15}: {v}")


if __name__ == "__main__":
    main()
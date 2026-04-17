#!/usr/bin/env python3
import argparse
import pandas as pd
import optuna
import train_model
import numpy as np

DATA_PATH = "pareto/SRAM/SRAM_pareto.csv"
_df = pd.read_csv(DATA_PATH)

def objective(trial):
    args = train_model.parse_args(args=[])
    args.tech       = "SRAM"
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    args.n_blocks   = trial.suggest_int("n_blocks", 4, 16)
    args.lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    args.dropout    = trial.suggest_float("dropout", 0.05, 0.2)
    args.alpha      = [trial.suggest_float("alpha_lat", 1.0, 15.0),
                       trial.suggest_float("alpha_area", 1.0, 15.0),
                       trial.suggest_float("alpha_energy", 1.0, 15.0),
                       trial.suggest_float("alpha_leak", 1.0, 15.0)]
    args.output_dir = f"model_output/tuning/trial_{trial.number}"
    args.data       = DATA_PATH
    args.sample_size = 30000

    _, _, _, metrics, y_true, y_pred = train_model.train(args, trial=trial)  

    msle_lat  = np.mean((np.log1p(y_true["Latency (ns)"]) - np.log1p(y_pred["Latency (ns)"]))**2)
    msle_area = np.mean((np.log1p(y_true["Area (mm^2)"])  - np.log1p(y_pred["Area (mm^2)"]))**2)
    msle_en   = np.mean((np.log1p(y_true["Energy (nJ)"])  - np.log1p(y_pred["Energy (nJ)"]))**2)
    msle_leak = np.mean((np.log1p(y_true["Leakage (mW)"]) - np.log1p(y_pred["Leakage (mW)"]))**2)

    # MAE and MedRelErr
    trial.set_user_attr("lat_mae",    metrics.get("Latency (ns)_MAE", 0))
    trial.set_user_attr("area_mae",   metrics.get("Area (mm^2)_MAE", 0))
    trial.set_user_attr("en_mae",     metrics.get("Energy (nJ)_MAE", 0))
    trial.set_user_attr("leak_mae",   metrics.get("Leakage (mW)_MAE", 0))
    
    trial.set_user_attr("lat_medrel", metrics.get("Latency (ns)_MedRelErr_percent", 0))
    trial.set_user_attr("area_medrel",metrics.get("Area (mm^2)_MedRelErr_percent", 0))
    trial.set_user_attr("en_medrel",  metrics.get("Energy (nJ)_MedRelErr_percent", 0))
    trial.set_user_attr("leak_medrel",metrics.get("Leakage (mW)_MedRelErr_percent", 0))

    return msle_lat + msle_area + msle_en + msle_leak  # unit-free

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",     type=int, default=10)
    parser.add_argument("--study-name", default="sram_opt_v1")
    parser.add_argument("--db",         default="sqlite:///optuna_study.db")
    parser.add_argument("--init",       action="store_true")
    args = parser.parse_args()

    study = optuna.create_study(study_name=args.study_name, storage=args.db,
                                load_if_exists=True, direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))
    if not args.init:
        study.optimize(objective, n_trials=args.trials)

if __name__ == "__main__":
    main()
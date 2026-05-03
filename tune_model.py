#!/usr/bin/env python3
import os, sys, optuna, train_model

# ── Configuration ────────────────────────────────────────────
TECH        = "SRAM"
ARCH        = True
TRIALS      = 30
SAMPLE_SIZE = 50000
EPOCHS      = 300
PATIENCE    = 50
STUDY_NAME  = "destiny_tune_v2"
# ─────────────────────────────────────────────────────────────

suffix    = "_arch" if ARCH else ""
data_path = f"pareto/{TECH}{suffix}/{TECH}{suffix}_full_data.csv"

def objective(trial):
    args = train_model.parse_args(args=[])
    args.tech       = TECH
    args.data       = data_path
    args.sample_size = SAMPLE_SIZE
    args.epochs     = EPOCHS
    args.patience   = PATIENCE
    args.output_dir = f"model_output/tuning/{STUDY_NAME}_trial_{trial.number}"
    args.eval_on_test = False

    args.hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    args.n_blocks   = trial.suggest_int("n_blocks", 2, 10)
    args.lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    args.dropout    = trial.suggest_float("dropout", 0.2, 0.5)
    args.alpha      = [
        trial.suggest_float("alpha_lat",    1.0, 4.0),
        trial.suggest_float("alpha_energy", 1.0, 4.0),
        trial.suggest_float("alpha_area",   1.0, 5.0),
        trial.suggest_float("alpha_leak",   2.0, 10.0),
    ]

    _, _, _, metrics, _, _ = train_model.train(args, trial=trial)

    targets = ["Read Latency (ns)", "Write Energy (nJ)", "Area (mm^2)", "Leakage (mW)"]
    total   = sum(metrics[f"{t}_Log10_MSE"] for t in targets)

    for t in targets:
        trial.set_user_attr(f"val_log_mse_{t}", metrics[f"{t}_Log10_MSE"])

    return total

if not os.path.exists(data_path):
    sys.exit(f"ERROR: Data not found: {data_path}")

study = optuna.create_study(
    study_name = STUDY_NAME,
    storage    = "sqlite:///optuna_study.db",
    load_if_exists = True,
    direction  = "minimize",
    pruner     = optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=10),
)

study.optimize(objective, n_trials=TRIALS)

print(f"\nBest Loss: {study.best_value:.6f} | Trial: {study.best_trial.number}")
for k, v in study.best_params.items():
    print(f"  {k:15}: {v}")
#!/usr/bin/env python3
import os, sys, argparse, optuna, train_model

def parse_args():
    p = argparse.ArgumentParser(description="Tune DESTINY PPA surrogate model hyperparameters")
    p.add_argument("--tech",        default="SRAM")
    p.add_argument("--arch",        action="store_true")
    p.add_argument("--trials",      type=int,   default=30)
    p.add_argument("--sample-size", type=int,   default=50000)
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--patience",    type=int,   default=50)
    p.add_argument("--study-name",  default="destiny_tune")
    return p.parse_args()

def objective(trial, args):
    # Base training arguments
    train_args = train_model.parse_args(args=[])
    train_args.tech        = args.tech
    train_args.arch        = args.arch
    train_args.data        = f"pareto/{args.tech}{'_arch' if args.arch else ''}/{args.tech}{'_arch' if args.arch else ''}_full_data.csv"
    train_args.sample_size = args.sample_size
    train_args.epochs      = args.epochs
    train_args.patience    = args.patience
    train_args.output_dir  = f"model_output/tuning/{args.study_name}_trial_{trial.number}"
    train_args.eval_on_test = False

    # Suggested hyperparameters
    train_args.hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    train_args.n_blocks   = trial.suggest_int("n_blocks", 2, 10)
    train_args.lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    train_args.dropout    = trial.suggest_float("dropout", 0.2, 0.5)
    train_args.alpha      = [
        trial.suggest_float("alpha_lat",    1.0, 10.0),
        trial.suggest_float("alpha_energy", 1.0, 4.0),
        trial.suggest_float("alpha_area",   1.0, 5.0),
        trial.suggest_float("alpha_leak",   1.0, 5.0),
    ]

    _, _, _, metrics, _, _ = train_model.train(train_args, trial=trial)

    targets = ["Read Latency (ns)", "Write Energy (nJ)", "Area (mm^2)", "Leakage (mW)"]
    total   = sum(metrics[f"{t}_Log10_MSE"] for t in targets)

    for t in targets:
        trial.set_user_attr(f"val_log_mse_{t}", metrics[f"{t}_Log10_MSE"])

    return total

if __name__ == "__main__":
    args = parse_args()
    
    suffix    = "_arch" if args.arch else ""
    data_path = f"pareto/{args.tech}{suffix}/{args.tech}{suffix}_full_data.csv"
    
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: Data not found: {data_path}")

    study = optuna.create_study(
        study_name = args.study_name,
        storage    = "sqlite:///optuna_study.db",
        load_if_exists = True,
        direction  = "minimize",
        pruner     = optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=10),
    )

    study.optimize(lambda t: objective(t, args), n_trials=args.trials)

    print(f"\nBest Loss: {study.best_value:.6f} | Trial: {study.best_trial.number}")
    for k, v in study.best_params.items():
        print(f"  {k:15}: {v}")
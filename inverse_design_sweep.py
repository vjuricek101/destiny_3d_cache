import os, sys, argparse, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_design import InverseOptimizer
from destiny_utils import pareto_frontier_2d, pareto_step_line, cap_colormap, format_log_axis, add_cap_colorbar

warnings.filterwarnings("ignore", category=UserWarning)

# ── Metric metadata ───────────────────────────────────────────────────────────

METRIC_META = {
    "cache_hit_latency_ns":   {"label": "Read Latency (ns)",  "unit": "ns"},
    "cache_area_mm2":         {"label": "Area (mm²)",          "unit": "mm²"},
    "cache_hit_energy_nJ":    {"label": "Hit Energy (nJ)",     "unit": "nJ"},
    "cache_leakage_mW":       {"label": "Leakage (mW)",        "unit": "mW"},
    "cache_miss_latency_ns":  {"label": "Miss Latency (ns)",   "unit": "ns"},
    "cache_write_latency_ns": {"label": "Write Latency (ns)",  "unit": "ns"},
    "cache_write_energy_nJ":  {"label": "Write Energy (nJ)",   "unit": "nJ"},
}

# CSV column → PPA label returned by InverseOptimizer
METRIC_TO_PPA_LABEL = {
    "cache_hit_latency_ns": "Latency",
    "cache_area_mm2":       "Area",
    "cache_write_energy_nJ": "Energy",
    "cache_leakage_mW":     "Leakage",
}

# ── Target-point selection ────────────────────────────────────────────────────

def select_pareto(df, x_col, y_col):
    """Return the non-dominated Pareto front for x_col × y_col."""
    result = df[pareto_frontier_2d(df[x_col].values, df[y_col].values)].reset_index(drop=True)
    print(f"   Pareto-optimal points: {len(result)}")
    return result


def select_median(df, x_col, y_col, n_bins):
    """Return one median-y representative per equal-quantile x bin."""
    work  = df.copy().reset_index(drop=True)
    x_vals = work[x_col].values
    edges  = np.unique(np.percentile(x_vals, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 2:
        raise ValueError("Too few unique x values — reduce --n-bins.")

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask   = (x_vals >= lo) & (x_vals <= hi if hi == edges[-1] else x_vals < hi)
        bucket = work[mask]
        if len(bucket):
            rows.append(work.loc[(bucket[y_col] - bucket[y_col].median()).abs().idxmin()])

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f"   Median points selected: {len(result)} (from {n_bins} requested bins)")
    return result

# ── DESTINY validation ────────────────────────────────────────────────────────

def validate_and_capture(tech, cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage,
                          node=32, roadmap="HP", timeout=60, is_arch=False):
    """Run DESTINY with the given design parameters; return PPA dict or None."""
    if is_arch:
        cell_content = f"""
-MemCellType: {tech}
-CellArea (F^2): 180.0000
-SRAMCellNMOSWidth (F): 2.5000
-SRAMCellPMOSWidth (F): 2.0000
-AccessCMOSWidth (F): 2.5000
-AccessType: CMOS
-MinSenseVoltage (mV): 32.0000
-CellAspectRatio: 1.5000
-ReadVoltage (V): 1.1
-Stitching: 16
-ProcessNode: {node}
"""
    else:
        cell_area = 60 + 20 * (wn + wac) + 10 * wp
        min_sense = 80 / wac
        cell_content = f"""
-MemCellType: SRAM
-CellArea (F^2): {cell_area:.4f}
-SRAMCellNMOSWidth (F): {wn:.4f}
-SRAMCellPMOSWidth (F): {wp:.4f}
-AccessCMOSWidth (F): {wac:.4f}
-AccessType: CMOS
-MinSenseVoltage (mV): {min_sense:.4f}
-ReadVoltage (V): {read_voltage:.4f}
-ProcessNode: {node}
"""

    cfg_content = f"""
-OptimizationTarget: Full
-Capacity (KB): {cap_kb}
-WordWidth (bit): {ww}
-Associativity (for cache only): {assoc}
-StackedDieCount: {stack}
-Temperature (K): {temp}
-DeviceRoadmap: {roadmap}
-MemoryCellInputFile: {os.path.abspath("validation_temp_bench.cell")}
"""
    cell_file, cfg_file, csv_file = (
        "validation_temp_bench.cell",
        "validation_temp_bench.cfg",
        "validation_temp_bench.csv",
    )
    try:
        with open(cell_file, "w") as f: f.write(cell_content)
        with open(cfg_file,  "w") as f: f.write(cfg_content)

        try:
            res = subprocess.run(["./destiny", cfg_file],
                                 capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  [warn] DESTINY timed out after {timeout}s — skipping")
            return None

        if res.returncode != 0 or not os.path.exists(csv_file):
            return None

        df = pd.read_csv(csv_file)
        return {col: float(df[col].iloc[0]) for col in [
            "cache_hit_latency_ns", "cache_area_mm2",
            "cache_write_energy_nJ",  "cache_leakage_mW",
        ]}
    except Exception as e:
        print(f"  [warn] DESTINY run failed: {e}"); return None
    finally:
        for f in [cell_file, cfg_file, csv_file]:
            if os.path.exists(f): os.remove(f)

# ── Helpers ───────────────────────────────────────────────────────────────────

def row_to_context(row, roadmap):
    """Extract fixed physical context from a dataset row for the optimizer."""
    node = int(row["process_node_nm"])
    ctx  = {f"process_node_nm_{node}": 1.0, "temperature_K": float(row.get("temperature_K", 350.0))}
    for rm in ["HP", "LOP", "LSTP"]:
        ctx[f"device_roadmap_{rm}"] = 1.0 if rm == roadmap else 0.0
    # Prefixed _ so they're stripped before the optimizer call but kept for DESTINY
    ctx["_wn"]   = float(row.get("CellInput_SRAMCellNMOSWidth (F)", 2.5))
    ctx["_wp"]   = float(row.get("CellInput_SRAMCellPMOSWidth (F)", 2.0))
    ctx["_wac"]  = float(row.get("CellInput_AccessCMOSWidth (F)",   2.5))
    ctx["_read_voltage"] = float(row.get("CellInput_ReadVoltage (V)", 1.0))
    ctx["_temp"] = int(row.get("temperature_K", 350))
    return ctx

def pct_err(predicted, target):
    """Signed percentage error: (predicted − target) / |target| × 100."""
    return float("nan") if target == 0 else (predicted - target) / abs(target) * 100.0

def abs_pct_err(predicted, target):
    return abs(pct_err(predicted, target))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Benchmark InverseOptimizer against Pareto or median points.")
    p.add_argument("--tech",             default="SRAM")
    p.add_argument("--arch",             action="store_true",   help="Use architectural-only model")
    p.add_argument("--node",             type=int, default=32,  help="Process node [nm]")
    p.add_argument("--roadmap",          default="HP",          choices=["HP", "LOP", "LSTP"])
    p.add_argument("--mode",             default="pareto",      choices=["pareto", "median"])
    p.add_argument("--n-bins",           type=int, default=20,  help="[median] x-quantile bins")
    p.add_argument("--x-metric",         default="cache_write_energy_nJ", choices=list(METRIC_META))
    p.add_argument("--y-metric",         default="cache_area_mm2",       choices=list(METRIC_META))
    p.add_argument("--validate-top",     type=int, default=5,   help="Top-N designs to DESTINY-validate")
    p.add_argument("--opt-steps",        type=int, default=400, help="Gradient steps per optimization")
    p.add_argument("--destiny-timeout",  type=int, default=60,  help="Timeout [s] per DESTINY call")
    p.add_argument("--output-dir",       default="benchmark_results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    x_col, y_col   = args.x_metric, args.y_metric
    x_meta, y_meta = METRIC_META[x_col], METRIC_META[y_col]
    x_label, x_unit = x_meta["label"], x_meta["unit"]
    y_label, y_unit = y_meta["label"], y_meta["unit"]

    folder   = f"{args.tech}_arch" if args.arch else args.tech
    data_csv = os.path.join("pareto", folder,
                            f"{folder}_{'pareto' if args.mode == 'pareto' else 'full_data'}.csv")

    # ── 1. Load and filter ───────────────────────────────────────────────────
    mode_label = args.mode.capitalize()
    print(f"\n{'═'*70}")
    print(f"  Benchmark [{mode_label}]: {args.tech} | {args.node}nm | {args.roadmap}"
          + (" [arch]" if args.arch else ""))
    print(f"  X: {x_label}   Y: {y_label}   Dataset: {data_csv}")
    print(f"{'═'*70}\n")

    print(f"[1/6] Loading dataset: {data_csv}")
    df_all = pd.read_csv(data_csv)

    df = df_all[df_all["mem_cell_type"].str.upper() == args.tech.upper()].copy()
    df = df[df["process_node_nm"] == args.node]
    df = df[df["device_roadmap"].str.upper() == args.roadmap.upper()]
    df = df[
        (df["cache_hit_latency_ns"] < 100) & (df["cache_area_mm2"] < 1000) &
        (df["cache_write_energy_nJ"]  < 1000) & (df["cache_leakage_mW"] > 0) &
        (df["cache_leakage_mW"] < 1e7) & (df[x_col] > 0) & (df[y_col] > 0)
    ]
    print(f"   Rows after filter: {len(df):,}")
    if len(df) == 0: sys.exit("ERROR: No rows match the filter.")

    # ── 2. Select target points ──────────────────────────────────────────────
    print(f"\n[2/6] {'Extracting Pareto front' if args.mode == 'pareto' else f'Extracting median points ({args.n_bins} x-bins)'} on {x_col} × {y_col}")
    if args.mode == "pareto":
        df_target     = select_pareto(df, x_col, y_col)
        df_pareto_ref = df_target
    else:
        df_target     = select_median(df, x_col, y_col, n_bins=args.n_bins)
        df_pareto_ref = df[pareto_frontier_2d(df[x_col].values, df[y_col].values)]

    # ── 3. Load optimizer ────────────────────────────────────────────────────
    print(f"\n[3/6] Loading InverseOptimizer (tech={args.tech}, arch={args.arch})")
    opt = InverseOptimizer(args.tech, is_arch=args.arch)

    # ── 4. Optimize toward each target ───────────────────────────────────────
    print(f"\n[4/6] Running inverse optimization for {len(df_target)} targets "
          f"({args.opt_steps} steps each)…\n")

    records = []
    for i, row in df_target.iterrows():
        target_x, target_y = row[x_col], row[y_col]

        targets = {k: row[k] for k in [x_col, y_col] if k in METRIC_TO_PPA_LABEL}
        ctx     = row_to_context(row, args.roadmap)
        opt_ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}

        design, ppa = opt.optimize(targets, opt_ctx, steps=args.opt_steps)

        surr_x = ppa.get(METRIC_TO_PPA_LABEL.get(x_col, ""))
        surr_y = ppa.get(METRIC_TO_PPA_LABEL.get(y_col, ""))
        errs   = [abs_pct_err(v, t) for v, t in [(surr_x, target_x), (surr_y, target_y)] if v is not None]
        surr_mean_err = float(np.mean(errs)) if errs else float("nan")

        x_pct = float((df[x_col] <= target_x).mean() * 100)
        y_pct = float((df[y_col] <= target_y).mean() * 100)

        records.append({
            "target_idx":             i,
            "x_percentile":           x_pct,
            "y_percentile":           y_pct,
            "target_x":               target_x,   "target_y":               target_y,
            "surr_x":                 surr_x,      "surr_y":                 surr_y,
            "surr_err_x_pct":         pct_err(surr_x, target_x) if surr_x is not None else None,
            "surr_err_y_pct":         pct_err(surr_y, target_y) if surr_y is not None else None,
            "surr_mean_abs_err_pct":  surr_mean_err,
            "capacity_kb":            design.get("capacity_kb"),
            "word_width_bits":        design.get("word_width_bits"),
            "associativity":          design.get("associativity"),
            "data_stacked_die_count": design.get("data_stacked_die_count"),
            "_wn": ctx["_wn"], "_wp": ctx["_wp"], "_wac": ctx["_wac"],
            "_read_voltage": ctx["_read_voltage"], "_temp": ctx["_temp"],
            "destiny_x": None, "destiny_y": None,
            "destiny_err_x_pct": None, "destiny_err_y_pct": None,
            "destiny_mean_abs_err_pct": None,
        })

        print(f"  [{i+1:3d}/{len(df_target)}] "
              f"Target: {x_label.split()[0]}={target_x:.4g}{x_unit} (p{x_pct:.0f}), "
              f"{y_label.split()[0]}={target_y:.4g}{y_unit} (p{y_pct:.0f})  →  "
              f"Surr: {surr_x:.4g}, {surr_y:.4g}  (err={surr_mean_err:.1f}%)")

    df_res = pd.DataFrame(records)

    # ── 5. DESTINY validation for top-N ──────────────────────────────────────
    n_validate = min(args.validate_top, len(df_res))
    if n_validate > 0:
        print(f"\n[5/6] DESTINY validation for top-{n_validate} designs "
              f"(ranked by surrogate error)…\n")
        for rank, ri in enumerate(df_res["surr_mean_abs_err_pct"].sort_values().head(n_validate).index, 1):
            row = df_res.loc[ri]
            print(f"  Validating design #{rank}  "
                  f"(target_idx={int(row.target_idx)}, surr_err={row.surr_mean_abs_err_pct:.1f}%)")
            print(f"    Cap={row.capacity_kb}KB  WW={row.word_width_bits}  "
                  f"Assoc={row.associativity}  Stack={row.data_stacked_die_count}")

            destiny_ppa = validate_and_capture(
                tech=args.tech, cap_kb=int(row.capacity_kb), ww=int(row.word_width_bits),
                assoc=int(row.associativity), stack=int(row.data_stacked_die_count),
                temp=int(row["_temp"]), wn=float(row["_wn"]),
                wp=float(row["_wp"]), wac=float(row["_wac"]),
                read_voltage=float(row["_read_voltage"]),
                node=args.node, roadmap=args.roadmap,
                timeout=args.destiny_timeout, is_arch=args.arch,
            )
            if destiny_ppa is None:
                print(f"    ✗ DESTINY failed\n"); continue

            dx, dy   = destiny_ppa.get(x_col), destiny_ppa.get(y_col)
            dx_err   = pct_err(dx, row.target_x) if dx is not None else None
            dy_err   = pct_err(dy, row.target_y) if dy is not None else None
            d_mean   = np.mean([abs(e) for e in [dx_err, dy_err] if e is not None])

            df_res.at[ri, "destiny_x"]                = dx
            df_res.at[ri, "destiny_y"]                = dy
            df_res.at[ri, "destiny_err_x_pct"]        = dx_err
            df_res.at[ri, "destiny_err_y_pct"]        = dy_err
            df_res.at[ri, "destiny_mean_abs_err_pct"] = d_mean

            xk, yk = x_col.split("_")[2], y_col.split("_")[2]
            print(f"    DESTINY → {xk}={dx:.4g}, {yk}={dy:.4g}")
            print(f"    Errors:  Δ{xk}={dx_err:+.1f}%  Δ{yk}={dy_err:+.1f}%  mean={d_mean:.1f}%\n")
    else:
        print("\n[5/6] Skipping DESTINY validation (--validate-top 0)\n")

    # ── 6. Save CSV ───────────────────────────────────────────────────────────
    s_errs   = df_res["surr_mean_abs_err_pct"].dropna()
    d_errs   = df_res["destiny_mean_abs_err_pct"].dropna()
    csv_path = os.path.join(args.output_dir,
                            f"benchmark_{args.mode}_{args.tech}_{args.node}nm_{args.roadmap}.csv")
    df_res[[c for c in df_res.columns if not c.startswith("_")]].to_csv(csv_path, index=False)
    print(f"[6/6] Results saved → {csv_path}")
    print(f"      Surrogate mean |err|: {s_errs.mean():.2f}%  (median {s_errs.median():.2f}%,  n={len(s_errs)})")
    if len(d_errs) > 0:
        print(f"      DESTINY   mean |err|: {d_errs.mean():.2f}%  (median {d_errs.median():.2f}%,  n={len(d_errs)})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    norm, _ = cap_colormap(df["capacity_kb"])
    cmap    = plt.cm.viridis

    # Background cloud coloured by capacity
    ax.scatter(df[x_col], df[y_col], c=df["capacity_kb"], norm=norm, cmap=cmap,
               s=14, alpha=0.15, linewidths=0, zorder=1, label="All data")

    # Pareto reference line — prominent in pareto mode, dimmed in median mode
    pf_alpha = 0.85 if args.mode == "pareto" else 0.35
    pf_label = "Pareto front (target)" if args.mode == "pareto" else "Pareto front (ref)"
    sx_p, sy_p = pareto_step_line(df_pareto_ref[x_col].values, df_pareto_ref[y_col].values)
    ax.plot(sx_p, sy_p, color="#aaaaaa",
            lw=1.5 if args.mode == "pareto" else 1.0,
            ls="-" if args.mode == "pareto" else "--",
            zorder=2, alpha=pf_alpha)
    ax.scatter(df_pareto_ref[x_col], df_pareto_ref[y_col],
               c=df_pareto_ref["capacity_kb"], norm=norm, cmap=cmap,
               s=55 if args.mode == "pareto" else 28,
               marker="D", edgecolors="k", linewidths=0.5,
               zorder=3, alpha=pf_alpha, label=pf_label)

    # Median target points (median mode only)
    if args.mode == "median":
        ax.scatter(df_target[x_col], df_target[y_col],
                   facecolors="none", s=70, marker="s", edgecolors="k",
                   linewidths=1.5, zorder=4, label="Median sample (target)")

    # Surrogate predictions + connector lines
    mask_surr = df_res["surr_x"].notna() & df_res["surr_y"].notna()
    ax.scatter(df_res.loc[mask_surr, "surr_x"], df_res.loc[mask_surr, "surr_y"],
               c="#ffa657", s=60, marker="o", edgecolors="k",
               linewidths=0.5, zorder=5, alpha=0.85, label="Surrogate prediction")
    for _, r in df_res[mask_surr].iterrows():
        ax.plot([r.target_x, r.surr_x], [r.target_y, r.surr_y],
                color="#ffa657", lw=0.8, alpha=0.5, zorder=3)

    # DESTINY-validated points + dashed connector lines
    mask_dest = df_res["destiny_x"].notna() & df_res["destiny_y"].notna()
    if mask_dest.any():
        ax.scatter(df_res.loc[mask_dest, "destiny_x"], df_res.loc[mask_dest, "destiny_y"],
                   c="#3fb950", s=90, marker="*", edgecolors="k",
                   linewidths=0.5, zorder=6, label="DESTINY validated")
        for _, r in df_res[mask_dest].iterrows():
            ax.plot([r.target_x, r.destiny_x], [r.target_y, r.destiny_y],
                    color="#3fb950", lw=1.0, alpha=0.6, zorder=4, linestyle="--")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"{x_label} (log scale)", fontsize=10)
    ax.set_ylabel(f"{y_label} (log scale)", fontsize=10)
    format_log_axis(ax, axis="both")

    subtitle = (f"{len(df_target)} targets, {args.n_bins} bins"
                if args.mode == "median" else f"{len(df_target)} Pareto points")
    ax.set_title(
        f"Inverse Optimizer vs {mode_label} Points  [{subtitle}]\n"
        f"{args.tech} | {args.node}nm | {args.roadmap}" + (" [arch]" if args.arch else ""),
        fontsize=12, fontweight="bold", pad=10)

    ann_lines = [f"Surrogate  mean |err|: {s_errs.mean():.1f}% (n={len(s_errs)})"]
    if len(d_errs) > 0:
        ann_lines.append(f"DESTINY    mean |err|: {d_errs.mean():.1f}% (n={len(d_errs)})")
    ax.text(0.02, 0.97, "\n".join(ann_lines), transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa", alpha=0.9))

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    add_cap_colorbar(fig, [ax], norm)

    png_path = os.path.join(args.output_dir,
                            f"benchmark_{args.mode}_{args.tech}_{args.node}nm_{args.roadmap}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"           Plot saved → {png_path}\n")


if __name__ == "__main__":
    main()
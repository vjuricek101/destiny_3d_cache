import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from destiny_utils import (
    METRICS, LAT_COL, CAP_COL, TECH_COL, TECHS,
    TECH_MARKERS, TECH_COLORS, SUB_MARKERS, CAP_KB_LABELS,
    pareto_frontier_2d, pareto_step_line,
    cap_colormap, add_cap_colorbar, format_log_axis, power2_xticks,
)

def plot_tech_3panel(tech: str, df_pareto: pd.DataFrame, out_dir: str):
    """
    3-panel figure: Write EDP, Area, Leakage vs Latency.
    Points colored by capacity with a shared colorbar to the right.
    A per-capacity Pareto frontier line is drawn for each capacity tier.
    If multiple AccessTypes exist, creates multiple rows filtering by type.
    """
    df = df_pareto[df_pareto[TECH_COL] == tech].copy()
    if df.empty:
        return

    norm, _ = cap_colormap(df[CAP_COL])
    cmap = plt.cm.viridis
    caps_sorted = sorted(df[CAP_COL].unique())

    sub_col = None
    if "CellInput_AccessType" in df.columns and df["CellInput_AccessType"].nunique() > 1:
        sub_col = "CellInput_AccessType"
    elif "device_roadmap" in df.columns and df["device_roadmap"].nunique() > 1:
        sub_col = "device_roadmap"

    has_sub_col = sub_col is not None

    plot_metrics = [m for m in METRICS if m[0] != LAT_COL]
    n_cols = len(plot_metrics)

    if has_sub_col:
        sub_types = sorted(df[sub_col].dropna().unique())
        rows = 1 + len(sub_types)
        fig, axes = plt.subplots(rows, n_cols, figsize=(5.5 * n_cols, 4.5 * rows), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.5), constrained_layout=True)
        axes = np.array([axes])

    fig.suptitle(f"{tech} — Pareto Tradeoffs\n(single-mat configs excluded from plot)", fontsize=16 if has_sub_col else 14, fontweight="bold")

    for i in range(axes.shape[0]):
        if has_sub_col:
            if i == 0:
                df_row = df.copy()
                row_title = f"All {sub_col.replace('CellInput_', '')}s"
            else:
                st = sub_types[i - 1]
                df_row = df[df[sub_col] == st].copy()
                row_title = f"{sub_col.replace('CellInput_', '')}: {st}"
        else:
            df_row = df.copy()
            row_title = ""

        for j, (ycol, ylabel, yscale) in enumerate(plot_metrics):
            ax = axes[i, j]
            
            cols = list(set([LAT_COL, ycol, CAP_COL] + ([sub_col] if has_sub_col else [])))
            sub = df_row[cols].dropna(subset=[LAT_COL, ycol, CAP_COL])
            
            if sub.empty:
                ax.set_visible(False)
                continue

            x = sub[LAT_COL].values
            y = sub[ycol].values

            # Scatter: all points coloured by capacity
            if has_sub_col:
                for row_st in sorted(sub[sub_col].dropna().unique()):
                    st_mask = (sub[sub_col] == row_st).values
                    if st_mask.sum() == 0: continue
                    ax.scatter(x[st_mask], y[st_mask],
                               c=sub[CAP_COL].values[st_mask], norm=norm, cmap=cmap,
                               marker=SUB_MARKERS.get(row_st, "o"),
                               s=35, alpha=0.65, linewidths=0, zorder=3)
            else:
                ax.scatter(x, y,
                           c=sub[CAP_COL].values, norm=norm, cmap=cmap,
                           s=35, alpha=0.65, linewidths=0, zorder=3)

            # Per-capacity Pareto frontier lines
            for cap in caps_sorted:
                cap_mask = sub[CAP_COL].values == cap
                x_cap = x[cap_mask]
                y_cap = y[cap_mask]
                if len(x_cap) < 2:
                    continue
                pf_cap = pareto_frontier_2d(x_cap, y_cap)
                if pf_cap.sum() < 2:
                    continue
                lx, ly = pareto_step_line(x_cap[pf_cap], y_cap[pf_cap])
                cap_color = cmap(norm(cap))
                ax.plot(lx, ly, color=cap_color, linewidth=1.1,
                        alpha=0.85, zorder=5)

            ax.set_xscale("log")
            ax.set_yscale(yscale)
            ax.set_xlabel("Cache Hit Latency (ns) (log scale)", fontsize=10)
            ax.set_ylabel(f"{ylabel} (log scale)" if yscale == "log" else ylabel, fontsize=10)
            t_prefix = f"[{row_title}] " if row_title else ""
            ax.set_title(f"{t_prefix}{ylabel} vs Latency", fontsize=11)
            format_log_axis(ax, axis="both" if yscale == "log" else "x")
            ax.grid(True, which="both", alpha=0.3)

    # Colorbar to the right of the rightmost panel
    add_cap_colorbar(fig, list(axes.flatten()), norm)
    
    # Legend on the first axes
    if has_sub_col:
        handles = [Line2D([0],[0], marker=SUB_MARKERS.get(st, "o"), color="w", 
                          markerfacecolor="gray", markersize=7, label=st)
                   for st in sub_types]
        axes[0, 0].legend(handles=handles, title=sub_col.replace('CellInput_', ''), fontsize=8, loc="upper left")

    path = os.path.join(out_dir, "tradeoffs_3panel.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_dominated_vs_pareto(tech: str, df_full: pd.DataFrame,
                              df_pareto: pd.DataFrame, out_dir: str):
    """
    For each of the 3 metrics, plot:
      - Full design space colored by capacity at low alpha (faded)
      - Pareto-optimal points colored by capacity at full opacity (bold)
      - If multiple AccessTypes exist, create extra rows filtering the Pareto points by type.
    """
    df_f = df_full[df_full[TECH_COL] == tech].copy()
    df_p = df_pareto[df_pareto[TECH_COL] == tech].copy()
    if df_p.empty or df_f.empty:
        return

    all_caps = pd.concat([df_f[CAP_COL], df_p[CAP_COL]])
    norm = LogNorm(vmin=all_caps.min(), vmax=all_caps.max())
    cmap = plt.cm.viridis

    sub_col = None
    if "CellInput_AccessType" in df_p.columns and df_p["CellInput_AccessType"].nunique() > 1:
        sub_col = "CellInput_AccessType"
    elif "device_roadmap" in df_p.columns and df_p["device_roadmap"].nunique() > 1:
        sub_col = "device_roadmap"

    plot_metrics = [m for m in METRICS if m[0] != LAT_COL]
    n_cols = len(plot_metrics)

    has_sub_col = sub_col is not None
    if has_sub_col:
        sub_types = sorted(df_p[sub_col].dropna().unique())
        rows = 1 + len(sub_types)
        fig, axes = plt.subplots(rows, n_cols, figsize=(5.5 * n_cols, 4.5 * rows), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.5), constrained_layout=True)
        axes = np.array([axes])  # Make 2D for consistent indexing

    fig.suptitle(f"{tech} — Full Design Space vs Pareto Frontier\n(single-mat configs excluded from plot)", 
                 fontsize=16 if has_sub_col else 14, fontweight="bold")

    for i in range(axes.shape[0]):
        if has_sub_col:
            if i == 0:
                df_p_row = df_p.copy()
                df_f_row = df_f.copy()
                row_title = f"All {sub_col.replace('CellInput_', '')}s"
            else:
                st = sub_types[i - 1]
                df_p_row = df_p[df_p[sub_col] == st].copy()
                df_f_row = df_f[df_f[sub_col] == st].copy()
                row_title = f"{sub_col.replace('CellInput_', '')}: {st}"
        else:
            df_p_row = df_p.copy()
            df_f_row = df_f.copy()
            row_title = ""

        for j, (ycol, ylabel, yscale) in enumerate(plot_metrics):
            ax = axes[i, j]
            
            cols = list(set([LAT_COL, ycol, CAP_COL] + ([sub_col] if has_sub_col else [])))
            
            valid_f = df_f_row[cols].dropna(subset=[LAT_COL, ycol, CAP_COL])
            valid_p_row = df_p_row[cols].dropna(subset=[LAT_COL, ycol, CAP_COL])
            
            if valid_f.empty or valid_p_row.empty:
                ax.set_visible(False)
                continue

            # Dominated cloud: Full design space ALWAYS shown in background
            ax.scatter(valid_f[LAT_COL], valid_f[ycol],
                       c=valid_f[CAP_COL], norm=norm, cmap=cmap,
                       s=14, alpha=0.15, linewidths=0, zorder=1)

            # Pareto-optimal points for this row
            if has_sub_col:
                for row_st in sorted(valid_p_row[sub_col].dropna().unique()):
                    st_mask = valid_p_row[sub_col] == row_st
                    if st_mask.sum() == 0: continue
                    ax.scatter(valid_p_row[LAT_COL][st_mask], valid_p_row[ycol][st_mask],
                               c=valid_p_row[CAP_COL][st_mask], norm=norm, cmap=cmap,
                               marker=SUB_MARKERS.get(row_st, "o"),
                               s=65, alpha=0.95, edgecolors="k", linewidths=0.5, zorder=3)
            else:
                ax.scatter(valid_p_row[LAT_COL], valid_p_row[ycol],
                           c=valid_p_row[CAP_COL], norm=norm, cmap=cmap,
                           s=65, alpha=0.95, edgecolors="k", linewidths=0.5, zorder=3)

            ax.set_xscale("log")
            ax.set_yscale(yscale)
            ax.set_xlabel("Cache Hit Latency (ns) (log scale)", fontsize=10)
            ax.set_ylabel(f"{ylabel} (log scale)" if yscale == "log" else ylabel, fontsize=10)
            t_prefix = f"[{row_title}] " if row_title else ""
            ax.set_title(f"{t_prefix}{ylabel} vs Latency", fontsize=11)
            format_log_axis(ax, axis="both" if yscale == "log" else "x")
            ax.grid(True, which="both", alpha=0.3)

    # Shared colorbar
    add_cap_colorbar(fig, axes.flatten().tolist(), norm)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=7, alpha=0.5, label="Full design space"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=9, markeredgecolor="k", markeredgewidth=0.5,
               label="Pareto-optimal"),
    ]
    if has_sub_col:
        for st in sub_types:
            legend_handles.append(Line2D([0], [0], marker=SUB_MARKERS.get(st, "o"), 
                                         color="w", markerfacecolor="k", markersize=6, label=f"{sub_col.replace('CellInput_', '')}: {st}"))

    axes[0, 0].legend(handles=legend_handles, fontsize=8, loc="upper left")

    path = os.path.join(out_dir, "dominated_vs_pareto.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_global_comparison(df: pd.DataFrame, metric_col: str, ylabel: str,
                            yscale: str, out_path: str):
    """
    Single-metric global plot: 1x4 panels.
    Panel 0: All 3 technologies overlapping.
    Panels 1-3: SRAM, RRAM, eDRAM side-by-side.
    Color=capacity, shape=tech. Shared axes for true 1:1 visual comparison.
    Per-capacity Pareto frontier drawn as stepped colored lines.
    """
    df = df[[LAT_COL, metric_col, CAP_COL, TECH_COL]].dropna()
    if df.empty:
        return

    norm, _ = cap_colormap(df[CAP_COL])
    cmap = plt.cm.viridis
    caps_sorted = sorted(df[CAP_COL].unique())

    fig, axes = plt.subplots(1, 1 + len(TECHS), figsize=(22, 5.5), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(f"Global Technology Comparison — {ylabel} vs Hit Latency\n(single-mat configs excluded from plot)", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes):
        is_all_tech = (i == 0)
        target_techs = TECHS if is_all_tech else [TECHS[i - 1]]
        
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        if ax == axes[0]:
            ax.set_ylabel(f"{ylabel} (log scale)" if yscale == "log" else ylabel, fontsize=11)
        ax.set_xlabel("Cache Hit Latency (ns) (log scale)", fontsize=11)
        ax.set_title("All Technologies" if is_all_tech else target_techs[0], fontsize=13, fontweight="bold" if is_all_tech else "normal")
        format_log_axis(ax, axis="both" if yscale == "log" else "x")
        ax.grid(True, which="both", alpha=0.3)

        for tech in target_techs:
            sub = df[df[TECH_COL] == tech]
            if sub.empty:
                continue

            x = sub[LAT_COL].values
            y = sub[metric_col].values
            caps = sub[CAP_COL].values

            ax.scatter(x, y,
                       c=caps, norm=norm, cmap=cmap,
                       marker=TECH_MARKERS.get(tech, "o"),
                       s=55, alpha=0.7, linewidths=0.3,
                       edgecolors=TECH_COLORS.get(tech, "k"),
                       zorder=3)

            # Per-capacity Pareto frontier lines
            if not is_all_tech:
                for cap in caps_sorted:
                    cap_mask = caps == cap
                    x_cap = x[cap_mask]
                    y_cap = y[cap_mask]
                    if len(x_cap) < 2:
                        continue
                    pf_cap = pareto_frontier_2d(x_cap, y_cap)
                    if pf_cap.sum() < 2:
                        continue
                    lx, ly = pareto_step_line(x_cap[pf_cap], y_cap[pf_cap])
                    cap_color = cmap(norm(cap))
                    ax.plot(lx, ly, color=cap_color, linewidth=1.1,
                            alpha=0.85, zorder=5)

    add_cap_colorbar(fig, list(axes), norm)

    # Add a custom legend on the "All Technologies" panel (axes[0])
    handles = [Line2D([0], [0], marker=TECH_MARKERS.get(t, "o"), color="w", 
                      markerfacecolor=TECH_COLORS.get(t, "k"), markersize=8, label=t)
               for t in TECHS if t in df[TECH_COL].values]
    axes[0].legend(handles=handles, title="Technology", fontsize=9, loc="upper left")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_scaling_comparison(df: pd.DataFrame, out_path: str):
    """Min latency per capacity per technology — power-of-2 x-axis with crossover annotation."""
    best = df.groupby([TECH_COL, CAP_COL])[LAT_COL].min().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))

    for tech in TECHS:
        sub = best[best[TECH_COL] == tech].sort_values(CAP_COL)
        if sub.empty:
            continue
        ax.plot(sub[CAP_COL], sub[LAT_COL],
                marker=TECH_MARKERS[tech],
                color=TECH_COLORS[tech],
                linewidth=2.2, markersize=7, label=tech)

    ax.set_xscale("log", base=2)
    power2_xticks(ax, best[CAP_COL].unique())
    ax.set_xlabel("Cache Capacity (log₂ scale)", fontsize=11)
    ax.set_ylabel("Min Cache Hit Latency (ns)", fontsize=11)
    ax.set_title("Technology Scaling Comparison — Best Achievable Latency\n(single-mat configs excluded from plot)", fontsize=13,
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def save_summary_stats(df: pd.DataFrame, out_path: str):
    rows = []
    for tech in TECHS:
        sub = df[df[TECH_COL] == tech]
        if sub.empty:
            continue
        for col, label, _ in [(LAT_COL, "Latency (ns)", None)] + METRICS:
            vals = sub[col].dropna()
            rows.append({
                "Technology": tech,
                "Metric":     label,
                "Min":        vals.min(),
                "Median":     vals.median(),
                "Max":        vals.max(),
                "Count":      len(vals),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

def main(pareto_csv: str):
    if not os.path.exists(pareto_csv):
        print(f"ERROR: {pareto_csv} not found. Run pareto_analysis.py first.")
        return

    print(f"Loading Pareto dataset: {pareto_csv}")
    df_pareto = pd.read_csv(pareto_csv)
    
    # Filter out single-mat configs for plotting
    df_pareto = df_pareto[df_pareto["data_total_mats"] > 1].copy()

    out_root = "pareto/plots"
    os.makedirs(out_root, exist_ok=True)

    # Per-technology plots
    for tech in df_pareto[TECH_COL].unique():
        tech_dir = os.path.join(out_root, tech)
        os.makedirs(tech_dir, exist_ok=True)
        print(f"\n[{tech}] Generating per-technology plots...")

        # 3-panel Pareto tradeoffs (which will expand to multi-row if multiple access types exist)
        plot_tech_3panel(tech, df_pareto, tech_dir)

        # Dominated vs Pareto overlay (needs full_data.csv)
        full_csv = f"pareto/{tech}/{tech}_full_data.csv"
        if os.path.exists(full_csv):
            print(f"  Loading full data: {full_csv}")
            df_full = pd.read_csv(full_csv)
            df_full = df_full[df_full["data_total_mats"] > 1].copy()
            plot_dominated_vs_pareto(tech, df_full, df_pareto, tech_dir)
        else:
            print(f"  Skipping dominated overlay ({full_csv} not found). "
                  "Run: python pareto_analysis.py --only-full")

    # Global comparison plots
    print("\n[Global] Generating cross-technology comparison plots...")
    for col, label, scale in METRICS:
        if col == LAT_COL:
            continue
        safe = label.replace(" ", "_").replace("²", "2").replace("(", "").replace(")", "").replace("/", "_")
        plot_global_comparison(
            df_pareto, col, label, scale,
            os.path.join(out_root, f"global_{safe}.png")
        )

    plot_scaling_comparison(df_pareto, os.path.join(out_root, "global_scaling_comparison.png"))

    # Summary stats
    save_summary_stats(df_pareto, os.path.join(out_root, "summary_stats.csv"))

    print("\nAll plots generated.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Pareto frontier plots.")
    p.add_argument("--pareto", default="pareto/pareto.csv",
                   help="Path to combined Pareto CSV.")
    args = p.parse_args()
    main(args.pareto)

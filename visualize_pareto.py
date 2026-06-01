## ## Configuration & Imports ####################################################
import argparse
import math
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
    TECH_SKIP_TARGETS,
    pareto_frontier_2d, pareto_step_line,
    cap_colormap, add_cap_colorbar, format_log_axis, power2_xticks,
)

## ## Layout Constants ###########################################################
# 4 panels per row keeps each panel ≥ 5.5 in wide on a ≤ 22 in figure.
# With 7 non-latency metrics the grid is 2 rows × 4 cols (last cell hidden).
_N_METRIC_COLS = 4


## ## Internal Helpers ###########################################################

def _build_subplot_grid(n_sub_groups: int, n_metrics: int):
    """
    Compute (total_rows, fig) parameters for a wrapping metric panel grid.

    Layout: n_sub_groups data slices (all, sub_type_1, …), each rendered as
    ceil(n_metrics / _N_METRIC_COLS) rows of up to _N_METRIC_COLS panels.
    Returns (n_metric_rows, total_rows).
    """
    n_metric_rows = math.ceil(n_metrics / _N_METRIC_COLS)
    total_rows = n_sub_groups * n_metric_rows
    return n_metric_rows, total_rows


def _get_ax(axes: np.ndarray, group_idx: int, metric_idx: int,
             n_metric_rows: int) -> plt.Axes:
    """Map (group, metric) → 2-D axes index within the wrapping grid."""
    row = group_idx * n_metric_rows + (metric_idx // _N_METRIC_COLS)
    col = metric_idx % _N_METRIC_COLS
    return axes[row, col]


def _hide_trailing_axes(axes: np.ndarray, n_sub_groups: int,
                        n_metrics: int, n_metric_rows: int) -> None:
    """Hide empty cells in the last metric row of every group."""
    remainder = n_metrics % _N_METRIC_COLS
    if remainder == 0:
        return
    for g in range(n_sub_groups):
        last_row = g * n_metric_rows + (n_metric_rows - 1)
        for c in range(remainder, _N_METRIC_COLS):
            axes[last_row, c].set_visible(False)


def _safe_yscale(ax: plt.Axes, y: np.ndarray, requested: str) -> str:
    """
    Apply y-axis scale, falling back to linear when log is requested but the
    metric has no positive finite values (e.g. refresh metrics on SRAM which
    the simulator reports as 0).  Returns the scale actually applied.
    """
    actual = requested
    if requested == "log" and not np.any(np.isfinite(y) & (y > 0)):
        actual = "linear"
    ax.set_yscale(actual)
    return actual


## ## Per-Technology Panel Plots #################################################

def plot_tech_panels(tech: str, df_pareto: pd.DataFrame, out_dir: str):
    """
    N-panel figure: all non-Read-Latency PPA metrics vs Read Latency (x-axis).
    Points are coloured by capacity; per-capacity Pareto frontier step-lines overlaid.
    If multiple AccessTypes or device roadmaps exist, a row group is added per sub-type.
    Grid: ceil(n_metrics / 4) rows × 4 cols per sub-type group.
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
    sub_types = sorted(df[sub_col].dropna().unique()) if has_sub_col else []
    n_sub_groups = 1 + len(sub_types) if has_sub_col else 1

    # Exclude metrics that are structurally zero for this technology (e.g. refresh for SRAM).
    skip = set(TECH_SKIP_TARGETS.get(tech, []))
    plot_metrics = [m for m in METRICS if m[0] != LAT_COL and m[0] not in skip]
    n_metrics = len(plot_metrics)
    n_metric_rows, total_rows = _build_subplot_grid(n_sub_groups, n_metrics)

    fig, axes = plt.subplots(
        total_rows, _N_METRIC_COLS,
        figsize=(5.5 * _N_METRIC_COLS, 4.5 * total_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    _hide_trailing_axes(axes, n_sub_groups, n_metrics, n_metric_rows)

    fig.suptitle(
        f"{tech} -- Pareto Tradeoffs\n(single-mat configs excluded from plot)",
        fontsize=16, fontweight="bold",
    )

    for g in range(n_sub_groups):
        if has_sub_col:
            if g == 0:
                df_row  = df.copy()
                row_title = f"All {sub_col.replace('CellInput_', '')}s"
            else:
                st        = sub_types[g - 1]
                df_row    = df[df[sub_col] == st].copy()
                row_title = f"{sub_col.replace('CellInput_', '')}: {st}"
        else:
            df_row    = df.copy()
            row_title = ""

        for m_idx, (ycol, ylabel, yscale) in enumerate(plot_metrics):
            ax   = _get_ax(axes, g, m_idx, n_metric_rows)
            cols = list(set([LAT_COL, ycol, CAP_COL] + ([sub_col] if has_sub_col else [])))
            sub  = df_row[cols].dropna(subset=[LAT_COL, ycol, CAP_COL])

            if sub.empty:
                ax.set_visible(False)
                continue

            x = sub[LAT_COL].values
            y = sub[ycol].values

            # Scatter: all points coloured by capacity
            if has_sub_col:
                for row_st in sorted(sub[sub_col].dropna().unique()):
                    st_mask = (sub[sub_col] == row_st).values
                    if st_mask.sum() == 0:
                        continue
                    ax.scatter(x[st_mask], y[st_mask],
                               c=sub[CAP_COL].values[st_mask], norm=norm, cmap=cmap,
                               marker=SUB_MARKERS.get(row_st, "o"),
                               s=35, alpha=0.65, linewidths=0, zorder=3)
            else:
                ax.scatter(x, y,
                           c=sub[CAP_COL].values, norm=norm, cmap=cmap,
                           s=35, alpha=0.65, linewidths=0, zorder=3)

            # Per-capacity Pareto step-line (lower capacity → faster read)
            for cap in caps_sorted:
                cap_mask = sub[CAP_COL].values == cap
                x_cap, y_cap = x[cap_mask], y[cap_mask]
                if len(x_cap) < 2:
                    continue
                pf = pareto_frontier_2d(x_cap, y_cap)
                if pf.sum() < 2:
                    continue
                lx, ly = pareto_step_line(x_cap[pf], y_cap[pf])
                ax.plot(lx, ly, color=cmap(norm(cap)), linewidth=1.1, alpha=0.85, zorder=5)

            ax.set_xscale("log")
            actual_yscale = _safe_yscale(ax, y, yscale)
            ax.set_xlabel("Read Latency (ns)", fontsize=10)
            ax.set_ylabel(f"{ylabel} (log)" if actual_yscale == "log" else ylabel, fontsize=10)
            t_prefix = f"[{row_title}] " if row_title else ""
            ax.set_title(f"{t_prefix}{ylabel} vs Read Latency", fontsize=10)
            format_log_axis(ax, axis="both" if actual_yscale == "log" else "x")
            ax.grid(True, which="both", alpha=0.3)

    add_cap_colorbar(fig, list(axes.flatten()), norm)

    if has_sub_col:
        handles = [
            Line2D([0], [0], marker=SUB_MARKERS.get(st, "o"), color="w",
                   markerfacecolor="gray", markersize=7, label=st)
            for st in sub_types
        ]
        _get_ax(axes, 0, 0, n_metric_rows).legend(
            handles=handles, title=sub_col.replace("CellInput_", ""), fontsize=8, loc="upper left"
        )

    path = os.path.join(out_dir, "tradeoffs_panels.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


## ## Dominated vs Pareto Overlay ################################################

def plot_dominated_vs_pareto(tech: str, df_full: pd.DataFrame,
                              df_pareto: pd.DataFrame, out_dir: str):
    """
    For each non-latency PPA metric:
      - Full design space at low alpha (dominated cloud)
      - Pareto-optimal points at full opacity with black edge
    Grid: ceil(n_metrics / 4) rows × 4 cols per sub-type group, mirroring plot_tech_panels.
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

    has_sub_col  = sub_col is not None
    sub_types    = sorted(df_p[sub_col].dropna().unique()) if has_sub_col else []
    n_sub_groups = 1 + len(sub_types) if has_sub_col else 1

    # Exclude metrics that are structurally zero for this technology (e.g. refresh for SRAM).
    skip = set(TECH_SKIP_TARGETS.get(tech, []))
    plot_metrics = [m for m in METRICS if m[0] != LAT_COL and m[0] not in skip]
    n_metrics    = len(plot_metrics)
    n_metric_rows, total_rows = _build_subplot_grid(n_sub_groups, n_metrics)

    fig, axes = plt.subplots(
        total_rows, _N_METRIC_COLS,
        figsize=(5.5 * _N_METRIC_COLS, 4.5 * total_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    _hide_trailing_axes(axes, n_sub_groups, n_metrics, n_metric_rows)

    fig.suptitle(
        f"{tech} -- Full Design Space vs Pareto Frontier\n(single-mat configs excluded from plot)",
        fontsize=16, fontweight="bold",
    )

    for g in range(n_sub_groups):
        if has_sub_col:
            if g == 0:
                df_p_row  = df_p.copy()
                df_f_row  = df_f.copy()
                row_title = f"All {sub_col.replace('CellInput_', '')}s"
            else:
                st        = sub_types[g - 1]
                df_p_row  = df_p[df_p[sub_col] == st].copy()
                df_f_row  = df_f[df_f[sub_col] == st].copy()
                row_title = f"{sub_col.replace('CellInput_', '')}: {st}"
        else:
            df_p_row  = df_p.copy()
            df_f_row  = df_f.copy()
            row_title = ""

        for m_idx, (ycol, ylabel, yscale) in enumerate(plot_metrics):
            ax   = _get_ax(axes, g, m_idx, n_metric_rows)
            cols = list(set([LAT_COL, ycol, CAP_COL] + ([sub_col] if has_sub_col else [])))

            valid_f   = df_f_row[cols].dropna(subset=[LAT_COL, ycol, CAP_COL])
            valid_p   = df_p_row[cols].dropna(subset=[LAT_COL, ycol, CAP_COL])

            if valid_f.empty or valid_p.empty:
                ax.set_visible(False)
                continue

            # Dominated cloud (full space, muted)
            ax.scatter(valid_f[LAT_COL], valid_f[ycol],
                       c=valid_f[CAP_COL], norm=norm, cmap=cmap,
                       s=14, alpha=0.15, linewidths=0, zorder=1)

            # Pareto-optimal overlay
            if has_sub_col:
                for row_st in sorted(valid_p[sub_col].dropna().unique()):
                    st_mask = valid_p[sub_col] == row_st
                    if st_mask.sum() == 0:
                        continue
                    ax.scatter(valid_p[LAT_COL][st_mask], valid_p[ycol][st_mask],
                               c=valid_p[CAP_COL][st_mask], norm=norm, cmap=cmap,
                               marker=SUB_MARKERS.get(row_st, "o"),
                               s=65, alpha=0.95, edgecolors="k", linewidths=0.5, zorder=3)
            else:
                ax.scatter(valid_p[LAT_COL], valid_p[ycol],
                           c=valid_p[CAP_COL], norm=norm, cmap=cmap,
                           s=65, alpha=0.95, edgecolors="k", linewidths=0.5, zorder=3)

            ax.set_xscale("log")
            # Use the union of full-space and Pareto y-values to decide scale
            y_all = np.concatenate([valid_f[ycol].values, valid_p[ycol].values])
            actual_yscale = _safe_yscale(ax, y_all, yscale)
            ax.set_xlabel("Read Latency (ns)", fontsize=10)
            ax.set_ylabel(f"{ylabel} (log)" if actual_yscale == "log" else ylabel, fontsize=10)
            t_prefix = f"[{row_title}] " if row_title else ""
            ax.set_title(f"{t_prefix}{ylabel} vs Read Latency", fontsize=10)
            format_log_axis(ax, axis="both" if actual_yscale == "log" else "x")
            ax.grid(True, which="both", alpha=0.3)

    add_cap_colorbar(fig, axes.flatten().tolist(), norm)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=7, alpha=0.5, label="Full design space"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=9, markeredgecolor="k", markeredgewidth=0.5,
               label="Pareto-optimal"),
    ]
    if has_sub_col:
        for st in sub_types:
            legend_handles.append(
                Line2D([0], [0], marker=SUB_MARKERS.get(st, "o"), color="w",
                       markerfacecolor="k", markersize=6,
                       label=f"{sub_col.replace('CellInput_', '')}: {st}")
            )
    _get_ax(axes, 0, 0, n_metric_rows).legend(
        handles=legend_handles, fontsize=8, loc="upper left"
    )

    path = os.path.join(out_dir, "dominated_vs_pareto.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


## ## Global Cross-Technology Comparison #########################################

def plot_global_comparison(df: pd.DataFrame, metric_col: str, ylabel: str,
                            yscale: str, out_path: str):
    """
    Single-metric global plot: 1 × (1 + |TECHS|) panels.
    Panel 0: all technologies overlapping.
    Panels 1+: per-technology with per-capacity Pareto step-lines.
    Shared axes enforce a true 1:1 visual comparison across technologies.
    """
    df = df[[LAT_COL, metric_col, CAP_COL, TECH_COL]].dropna()
    if df.empty:
        return

    norm, _ = cap_colormap(df[CAP_COL])
    cmap = plt.cm.viridis
    caps_sorted = sorted(df[CAP_COL].unique())

    fig, axes = plt.subplots(
        1, 1 + len(TECHS),
        figsize=(22, 5.5),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(
        f"Global Technology Comparison -- {ylabel} vs Read Latency\n"
        "(single-mat configs excluded from plot)",
        fontsize=16, fontweight="bold",
    )

    for i, ax in enumerate(axes):
        is_all_tech  = (i == 0)
        target_techs = TECHS if is_all_tech else [TECHS[i - 1]]

        ax.set_xscale("log")
        ax.set_yscale(yscale)
        if i == 0:
            ax.set_ylabel(f"{ylabel} (log)" if yscale == "log" else ylabel, fontsize=11)
        ax.set_xlabel("Read Latency (ns)", fontsize=11)
        ax.set_title(
            "All Technologies" if is_all_tech else target_techs[0],
            fontsize=13, fontweight="bold" if is_all_tech else "normal",
        )
        format_log_axis(ax, axis="both" if yscale == "log" else "x")
        ax.grid(True, which="both", alpha=0.3)

        for tech in target_techs:
            sub = df[df[TECH_COL] == tech]
            if sub.empty:
                continue

            x    = sub[LAT_COL].values
            y    = sub[metric_col].values
            caps = sub[CAP_COL].values

            ax.scatter(x, y,
                       c=caps, norm=norm, cmap=cmap,
                       marker=TECH_MARKERS.get(tech, "o"),
                       s=55, alpha=0.7, linewidths=0.3,
                       edgecolors=TECH_COLORS.get(tech, "k"),
                       zorder=3)

            # Per-capacity Pareto step-lines for individual-tech panels only
            if not is_all_tech:
                for cap in caps_sorted:
                    cap_mask = caps == cap
                    x_cap, y_cap = x[cap_mask], y[cap_mask]
                    if len(x_cap) < 2:
                        continue
                    pf = pareto_frontier_2d(x_cap, y_cap)
                    if pf.sum() < 2:
                        continue
                    lx, ly = pareto_step_line(x_cap[pf], y_cap[pf])
                    ax.plot(lx, ly, color=cmap(norm(cap)), linewidth=1.1, alpha=0.85, zorder=5)

    add_cap_colorbar(fig, list(axes), norm)

    handles = [
        Line2D([0], [0], marker=TECH_MARKERS.get(t, "o"), color="w",
               markerfacecolor=TECH_COLORS.get(t, "k"), markersize=8, label=t)
        for t in TECHS if t in df[TECH_COL].values
    ]
    axes[0].legend(handles=handles, title="Technology", fontsize=9, loc="upper left")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


## ## Scaling Comparison #########################################################

def plot_scaling_comparison(df: pd.DataFrame, out_path: str):
    """Min read latency per capacity per technology -- power-of-2 x-axis with crossover annotation."""
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
    ax.set_xlabel("Cache Capacity (log2 scale)", fontsize=11)
    ax.set_ylabel("Min Read Latency (ns)", fontsize=11)
    ax.set_title(
        "Technology Scaling Comparison -- Best Achievable Read Latency\n"
        "(single-mat configs excluded from plot)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


## ## Summary Statistics #########################################################

def save_summary_stats(df: pd.DataFrame, out_path: str):
    """Per-technology min/median/max/count for all 8 PPA targets."""
    rows = []
    for tech in TECHS:
        sub = df[df[TECH_COL] == tech]
        if sub.empty:
            continue
        for col, label, _ in METRICS:
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


## ## Entry Point ################################################################

def main(pareto_csv: str):
    if not os.path.exists(pareto_csv):
        print(f"ERROR: {pareto_csv} not found. Run pareto_analysis.py first.")
        return

    print(f"Loading Pareto dataset: {pareto_csv}")
    df_pareto = pd.read_csv(pareto_csv)

    # Single-mat configs are degenerate (trivial area-latency trade-off suppressed)
    df_pareto = df_pareto[df_pareto["data_total_mats"] > 1].copy()

    out_root = "pareto/plots"
    os.makedirs(out_root, exist_ok=True)

    # Per-technology plots
    for tech in df_pareto[TECH_COL].unique():
        tech_dir = os.path.join(out_root, tech)
        os.makedirs(tech_dir, exist_ok=True)
        print(f"\n[{tech}] Generating per-technology plots...")

        plot_tech_panels(tech, df_pareto, tech_dir)

        full_csv = f"pareto/{tech}/{tech}_full_data.csv"
        if os.path.exists(full_csv):
            print(f"  Loading full data: {full_csv}")
            df_full = pd.read_csv(full_csv)
            df_full = df_full[df_full["data_total_mats"] > 1].copy()
            plot_dominated_vs_pareto(tech, df_full, df_pareto, tech_dir)
        else:
            print(f"  Skipping dominated overlay ({full_csv} not found). "
                  "Run: python pareto_analysis.py --only-full")

    # Global comparison plots -- one figure per non-latency metric
    print("\n[Global] Generating cross-technology comparison plots...")
    for col, label, scale in METRICS:
        if col == LAT_COL:
            continue
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        plot_global_comparison(
            df_pareto, col, label, scale,
            os.path.join(out_root, f"global_{safe}.png"),
        )

    plot_scaling_comparison(df_pareto, os.path.join(out_root, "global_scaling_comparison.png"))

    save_summary_stats(df_pareto, os.path.join(out_root, "summary_stats.csv"))

    print("\nAll plots generated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Pareto frontier plots.")
    p.add_argument("--pareto", default="pareto/pareto.csv",
                   help="Path to combined Pareto CSV.")
    args = p.parse_args()
    main(args.pareto)

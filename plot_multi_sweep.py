#!/usr/bin/env python3
"""
plot_multi_sweep.py
Loads the CSVs produced by inverse_design_multi_sweep.py and produces
two PNG figures (latency/area and energy/leakage families),
each a 4×3 grid of scatter panels.

Each panel shows four series in stacking order:
  1. Target points        (black squares)
  2. Pre-snap surrogate   (orange circles  + connector lines)
  3. Post-snap surrogate  (blue diamonds   + connector lines)
  4. DESTINY validated    (green stars     + dashed connector lines)  [if present]

A shared suptitle at the top of every figure shows technology, process node,
and device roadmap so the figure is self-contained.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from inverse_design import TARGET_KEY_TO_OPT_TARGET
from inverse_design_multi_sweep import METRIC_META

# ── Colour palette ──────────────────────────────────────────────────────────
C_TARGET    = "#1a1a2e"   # near-black
C_PRE_SNAP  = "#ffa657"   # orange
C_POST_SNAP = "#4d9de0"   # blue
C_DESTINY   = "#3fb950"   # green


def draw_missing(ax):
    ax.text(0.5, 0.5, "Missing", ha="center", va="center",
            transform=ax.transAxes, color="red", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _scatter_with_connectors(ax, src_x, src_y, dest_x, dest_y,
                               color, marker, marker_size, label,
                               zorder, lw=0.9, ls="-", alpha_line=0.55):
    """Scatter dest points and draw connectors from (src_x, src_y)."""
    ax.scatter(dest_x, dest_y,
               marker=marker, s=marker_size, c=color,
               edgecolors="k", linewidths=0.45,
               label=label, zorder=zorder)
    for i in range(len(dest_x)):
        ax.plot([src_x[i], dest_x[i]], [src_y[i], dest_y[i]],
                color=color, lw=lw, ls=ls, alpha=alpha_line, zorder=zorder - 1)


def _error_annotation(ax, values, label_prefix):
    """Return a short mean-|err| string for the panel subtitle, or ''."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return ""
    return f"{label_prefix}={np.mean(clean):.1f}%"


def plot_panel(ax, df, x_col, y_col, variant, feas):
    if df is None or df.empty:
        draw_missing(ax)
        return

    target_x = df["target_x"].values
    target_y = df["target_y"].values

    # ── 1. Targets ────────────────────────────────────────────────────────
    ax.scatter(target_x, target_y,
               marker="s", s=22, c=C_TARGET,
               label="Target", zorder=6)

    all_x = list(target_x)
    all_y = list(target_y)

    err_parts = []

    # ── 2. Pre-snap surrogate ─────────────────────────────────────────────
    pre_mask = df["surr_x"].notna() & df["surr_y"].notna()
    if pre_mask.any():
        sx = df.loc[pre_mask, "surr_x"].values
        sy = df.loc[pre_mask, "surr_y"].values
        tx = df.loc[pre_mask, "target_x"].values
        ty = df.loc[pre_mask, "target_y"].values
        _scatter_with_connectors(ax, tx, ty, sx, sy,
                                 C_PRE_SNAP, "o", 30,
                                 "Pre-snap surr.", zorder=7)
        all_x.extend(sx); all_y.extend(sy)
        # mean error annotation
        if "surr_mean_abs_err_pct" in df.columns:
            e = _error_annotation(ax, df.loc[pre_mask, "surr_mean_abs_err_pct"].values,
                                  "pre")
            if e: err_parts.append(e)

    # ── 3. Post-snap surrogate ────────────────────────────────────────────
    post_mask = (df["post_snap_surr_x"].notna() & df["post_snap_surr_y"].notna()
                 if "post_snap_surr_x" in df.columns else pd.Series(False, index=df.index))
    if post_mask.any():
        px = df.loc[post_mask, "post_snap_surr_x"].values
        py = df.loc[post_mask, "post_snap_surr_y"].values
        tx = df.loc[post_mask, "target_x"].values
        ty = df.loc[post_mask, "target_y"].values
        _scatter_with_connectors(ax, tx, ty, px, py,
                                 C_POST_SNAP, "D", 26,
                                 "Post-snap surr.", zorder=8,
                                 lw=0.9, ls="-")
        all_x.extend(px); all_y.extend(py)
        if "post_snap_surr_mean_abs_err_pct" in df.columns:
            e = _error_annotation(ax, df.loc[post_mask,
                                  "post_snap_surr_mean_abs_err_pct"].values, "post")
            if e: err_parts.append(e)

    # ── 4. DESTINY validated ──────────────────────────────────────────────
    if "destiny_x" in df.columns and "destiny_y" in df.columns:
        dest_mask = df["destiny_x"].notna() & df["destiny_y"].notna()
        if dest_mask.any():
            dx = df.loc[dest_mask, "destiny_x"].values
            dy = df.loc[dest_mask, "destiny_y"].values
            tx = df.loc[dest_mask, "target_x"].values
            ty = df.loc[dest_mask, "target_y"].values
            _scatter_with_connectors(ax, tx, ty, dx, dy,
                                     C_DESTINY, "*", 55,
                                     "DESTINY", zorder=9,
                                     lw=1.0, ls="--", alpha_line=0.70)
            all_x.extend(dx); all_y.extend(dy)
            if "destiny_mean_abs_err_pct" in df.columns:
                e = _error_annotation(ax, df.loc[dest_mask,
                                      "destiny_mean_abs_err_pct"].values, "sim")
                if e: err_parts.append(e)

    # ── Axis formatting ───────────────────────────────────────────────────
    all_x = np.array([v for v in all_x if v is not None and v > 0], dtype=float)
    all_y = np.array([v for v in all_y if v is not None and v > 0], dtype=float)
    if len(all_x) > 0 and len(all_y) > 0:
        def _padded_lim(vals, pad=0.12):
            lo, hi = np.log10(vals.min()), np.log10(vals.max())
            p = max((hi - lo) * pad, 0.08)
            return 10 ** (lo - p), 10 ** (hi + p)
        ax.set_xlim(*_padded_lim(all_x))
        ax.set_ylim(*_padded_lim(all_y))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=7)

    feas_str = "Feas" if feas else "No-feas"
    err_str  = "  |  " + "  ".join(err_parts) if err_parts else ""
    ax.set_title(f"{variant}  [{feas_str}]{err_str}",
                 fontsize=8, pad=3)


def _build_legend_handles():
    """Shared legend for all panels."""
    handles = [
        mlines.Line2D([], [], marker="s", color="w",  markerfacecolor=C_TARGET,
                      markersize=7, label="Target"),
        mlines.Line2D([], [], marker="o", color="w",  markerfacecolor=C_PRE_SNAP,
                      markeredgecolor="k", markeredgewidth=0.4,
                      markersize=7, label="Pre-snap surrogate"),
        mlines.Line2D([], [], marker="D", color="w",  markerfacecolor=C_POST_SNAP,
                      markeredgecolor="k", markeredgewidth=0.4,
                      markersize=7, label="Post-snap surrogate"),
        mlines.Line2D([], [], marker="*", color="w",  markerfacecolor=C_DESTINY,
                      markeredgecolor="k", markeredgewidth=0.4,
                      markersize=10, label="DESTINY validated"),
    ]
    return handles


def main():
    p = argparse.ArgumentParser(
        description="Plot multi-sweep inverse-design results as a 4×3 panel grid."
    )
    p.add_argument("--output-dir", default="benchmark_results")
    p.add_argument("--tech",    default="SRAM")
    p.add_argument("--node",    type=int, default=32)
    p.add_argument("--roadmap", default="HP")
    p.add_argument("--mode",    default="pareto")
    args = p.parse_args()

    variants = ["baseline", "ste", "gumbel"]

    families = {
        "latency_area":    ("cache_hit_latency_ns",  "cache_area_mm2"),
        "energy_leakage":  ("cache_write_energy_nJ", "cache_leakage_mW"),
    }

    header = f"{args.tech} | {args.node} nm | {args.roadmap}"

    for fam_name, (m1, m2) in families.items():
        # 4 row configs: both orderings × both feasibility flags
        rows_config = [
            (m1, m2, False),
            (m1, m2, True),
            (m2, m1, False),
            (m2, m1, True),
        ]

        fig, axes = plt.subplots(4, 3, figsize=(13, 15), constrained_layout=False)
        fig.subplots_adjust(top=0.91, hspace=0.42, wspace=0.30,
                            left=0.08, right=0.97, bottom=0.06)

        # ── Figure-level header ──────────────────────────────────────────
        fam_label = ("Read Latency vs Area" if fam_name == "latency_area"
                     else "Write Energy vs Leakage")
        fig.suptitle(
            f"{fam_label}\n{header}",
            fontsize=13, fontweight="bold", y=0.97,
        )

        # ── Column headers (variant names) ───────────────────────────────
        for col_idx, variant in enumerate(variants):
            axes[0, col_idx].set_title(
                f"{variant.capitalize()}",
                fontsize=10, fontweight="bold", pad=14,
                color="#333333",
            )

        for row_idx, (x_col, y_col, feas) in enumerate(rows_config):
            feas_tag  = "1" if feas else "0"
            x_label   = METRIC_META.get(x_col, {}).get("label", x_col)
            y_label   = METRIC_META.get(y_col, {}).get("label", y_col)
            feas_str  = "w/ feasibility" if feas else "no feasibility"

            for col_idx, variant in enumerate(variants):
                ax = axes[row_idx, col_idx]
                fname = (f"sweep_{args.tech}_{args.node}nm_{args.roadmap}_"
                         f"{variant}_{x_col}_vs_{y_col}_feas{feas_tag}.csv")
                fpath = os.path.join(args.output_dir, fname)

                df = None
                if os.path.exists(fpath):
                    try:
                        df = pd.read_csv(fpath)
                    except Exception:
                        df = None

                plot_panel(ax, df, x_col, y_col, variant, feas)

                # Y-axis label on left-most column
                if col_idx == 0:
                    ax.set_ylabel(f"{y_label}\n({feas_str})",
                                  fontsize=8, fontweight="bold", labelpad=4)

                # X-axis label on bottom-most row of each metric pair
                if row_idx in [1, 3]:
                    ax.set_xlabel(x_label, fontsize=8, fontweight="bold", labelpad=3)

        # ── Shared legend below the grid ─────────────────────────────────
        fig.legend(handles=_build_legend_handles(),
                   loc="lower center", ncol=4, fontsize=9,
                   frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.01))

        fig_path = os.path.join(args.output_dir, f"plot_{fam_name}.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()

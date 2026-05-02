import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from destiny_utils import (
    METRICS, LAT_COL, CAP_COL, TECH_COL, TECHS,
    TECH_MARKERS, TECH_COLORS, SUB_MARKERS, CAP_KB_LABELS,
    pareto_frontier_2d, pareto_step_line,
    cap_colormap, add_cap_colorbar, format_log_axis, power2_xticks,
)
from visualize_pareto import (
    plot_tech_3panel,
    plot_dominated_vs_pareto,
    plot_global_comparison,
    plot_scaling_comparison,
    save_summary_stats
)

def plot_arch_parameter_impact(tech: str, df: pd.DataFrame, out_dir: str):
    """
    For each arch param, plot 1x4 panels (Latency, Write Energy, Area, Leakage vs Param).
    Colored by capacity, shaped by device_roadmap. Both axes log scaled.
    """
    params = {
        "word_width_bits": "Word Width (bits)",
        "associativity": "Associativity",
        "data_stacked_die_count": "Stack Tiers",
        "temperature_K": "Temperature (K)",
        "process_node_nm": "Process Node (nm)"
    }
    
    if df.empty:
        return

    norm, _ = cap_colormap(df[CAP_COL])
    cmap = plt.cm.viridis
    
    for param_col, param_label in params.items():
        if param_col not in df.columns or df[param_col].nunique() <= 1:
            continue
            
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
        fig.suptitle(f"{tech} — Impact of {param_label} on Metrics\n(single-mat configs excluded from plot)", fontsize=16, fontweight="bold")
        
        for j, (ycol, ylabel, yscale) in enumerate(METRICS):
            ax = axes[j]
            
            valid_df = df.dropna(subset=[param_col, ycol, CAP_COL])
            if valid_df.empty:
                ax.set_visible(False)
                continue
                
            x = valid_df[param_col].values
            y = valid_df[ycol].values
            caps = valid_df[CAP_COL].values
            
            if "device_roadmap" in valid_df.columns:
                roadmaps = sorted(valid_df["device_roadmap"].dropna().unique())
                n_rm = len(roadmaps)
                # Jitter x multiplicatively (±8%) so vertical columns of points don't perfectly overlap
                multipliers = np.linspace(0.92, 1.08, n_rm) if n_rm > 1 else [1.0]
                
                for idx, rm in enumerate(roadmaps):
                    rm_mask = (valid_df["device_roadmap"] == rm).values
                    if not rm_mask.any(): continue
                    
                    x_jitter = x[rm_mask].astype(float) * multipliers[idx]
                    
                    ax.scatter(x_jitter, y[rm_mask],
                               c=caps[rm_mask], norm=norm, cmap=cmap,
                               marker=SUB_MARKERS.get(rm, "o"),
                               s=40, alpha=0.5, edgecolors="none", zorder=3)
            else:
                ax.scatter(x, y,
                           c=caps, norm=norm, cmap=cmap,
                           s=40, alpha=0.5, edgecolors="none", zorder=3)
            
            ax.set_xscale("log")
            ax.set_yscale("log")  # Enforce log-log scale
            ax.set_xlabel(f"{param_label} (log scale)", fontsize=11)
            ax.set_ylabel(f"{ylabel} (log scale)", fontsize=11)
            ax.set_title(f"{ylabel} vs {param_label}", fontsize=13)
            
            format_log_axis(ax, axis="both")
            ax.grid(True, which="both", alpha=0.3)
            
        add_cap_colorbar(fig, list(axes), norm)
        
        if "device_roadmap" in df.columns:
            roadmaps = df["device_roadmap"].dropna().unique()
            if len(roadmaps) > 0:
                handles = [Line2D([0], [0], marker=SUB_MARKERS.get(rm, "o"), color="w",
                                  markerfacecolor="gray", markersize=8, label=rm)
                           for rm in sorted(roadmaps)]
                axes[0].legend(handles=handles, title="Roadmap", fontsize=9, loc="best")
                
        out_path = os.path.join(out_dir, f"arch_impact_{param_col}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main(pareto_csv: str):
    if not os.path.exists(pareto_csv):
        print(f"ERROR: {pareto_csv} not found. Run pareto_analysis.py --arch first.")
        return

    print(f"Loading architectural Pareto dataset: {pareto_csv}")
    df_pareto = pd.read_csv(pareto_csv)
    
    # Filter out single-mat configs for plotting
    df_pareto = df_pareto[df_pareto["data_total_mats"] > 1].copy()

    out_root = "pareto/plots_arch"
    os.makedirs(out_root, exist_ok=True)

    # Per-technology plots
    for tech in df_pareto[TECH_COL].unique():
        tech_dir = os.path.join(out_root, tech)
        os.makedirs(tech_dir, exist_ok=True)
        print(f"\n[{tech}] Generating per-technology plots...")

        # 3-panel Pareto tradeoffs
        plot_tech_3panel(tech, df_pareto, tech_dir)

        # Dominated vs Pareto overlay (needs arch full_data.csv)
        full_csv = f"pareto/{tech}_arch/{tech}_arch_full_data.csv"
        df_full_for_param = None
        if os.path.exists(full_csv):
            print(f"  Loading full data: {full_csv}")
            df_full = pd.read_csv(full_csv)
            df_full = df_full[df_full["data_total_mats"] > 1].copy()
            plot_dominated_vs_pareto(tech, df_full, df_pareto, tech_dir)
            df_full_for_param = df_full
        else:
            print(f"  Skipping dominated overlay ({full_csv} not found). "
                  "Run: python pareto_analysis.py --arch")
            df_full_for_param = df_pareto

        plot_arch_parameter_impact(tech, df_full_for_param, tech_dir)

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

    print("\nAll architectural plots generated.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate architectural Pareto frontier plots.")
    p.add_argument("--pareto", default="pareto/pareto.csv",
                   help="Path to combined Pareto CSV (which should contain arch data).")
    args = p.parse_args()
    main(args.pareto)

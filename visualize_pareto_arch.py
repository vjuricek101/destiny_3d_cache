import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
})

METRICS = [
    ("Cache Area (mm^2)",         "Area",         "log"),
    ("Cache Hit Energy (nJ)",     "Energy",        "log"),
    ("Cache Leakage Power (mW)",  "Leakage Power", "log"),
]
LAT_COL   = "Cache Hit Latency (ns)"
CAP_COL   = "capacity_mb"
TECH_COL  = "memory_technology"
TECHS     = ["SRAM", "RRAM", "eDRAM"]

CAP_MB_LABELS = {
    0.001953125: "2KB",   0.00390625: "4KB",   0.0078125: "8KB",
    0.015625:   "16KB",   0.03125:   "32KB",   0.0625:   "64KB",
    0.125:     "128KB",   0.25:     "256KB",   0.5:     "512KB",
    1.0:         "1MB",   2.0:        "2MB",   4.0:       "4MB",
    8.0:         "8MB",  16.0:       "16MB",  32.0:      "32MB",
}

def pareto_frontier_2d(x, y):
    """Return mask of non-dominated points minimizing both x and y."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]: continue
        # A point is dominated if another point is <= in both and < in at least one
        dominated[i] = np.any((x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i])))
    return ~dominated

def pareto_step_line(x, y):
    """Create a staircase path for a 2D Pareto frontier."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    sx, sy = [xs[0]], [ys[0]]
    for xi, yi in zip(xs[1:], ys[1:]):
        sy.append(sy[-1])
        sx.append(xi)
        sx.append(xi)
        sy.append(yi)
    return np.array(sx), np.array(sy)

def plot_pareto_shift_matrix(tech: str, df: pd.DataFrame, out_dir: str):
    """
    A grid of plots showing how the Pareto frontier shifts per architectural knob.
    Rows: Architectural parameters (Assoc, WW, Stacking, Temp)
    Cols: Tradeoffs (Lat-Area, Lat-Energy, Lat-Leakage)
    """
    knobs = ["associativity", "word_width", "stacked_die_count", "Temperature (K)"]
    labels = ["Associativity", "Word Width", "Stacking Layers", "Temperature (K)"]
    
    # Filter for a representative fixed capacity (middle of the sweep)
    caps = sorted(df[CAP_COL].unique())
    target_cap = caps[len(caps)//2] 
    sub = df[df[CAP_COL] == target_cap].copy()
    
    rows, cols = len(knobs), len(METRICS)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), constrained_layout=True)
    fig.suptitle(f"Architectural Pareto Shift Matrix — {tech} (Capacity: {CAP_MB_LABELS.get(target_cap, target_cap)})", 
                 fontsize=18, fontweight="bold")

    for i, (knob, klabel) in enumerate(zip(knobs, labels)):
        if knob not in sub.columns:
            for j in range(cols): axes[i, j].set_visible(False)
            continue
            
        unique_vals = sorted(sub[knob].unique())
        norm = plt.Normalize(vmin=0, vmax=max(1, len(unique_vals)-1))
        cmap = plt.cm.plasma
        
        for j, (ycol, ylabel, yscale) in enumerate(METRICS):
            ax = axes[i, j]
            ax.set_xscale("log")
            ax.set_yscale(yscale)
            ax.set_title(f"{klabel} Shift: {ylabel} vs Latency")
            ax.set_xlabel("Latency (ns)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
            
            for v_idx, val in enumerate(unique_vals):
                val_sub = sub[sub[knob] == val].dropna(subset=[LAT_COL, ycol])
                if len(val_sub) < 2: continue
                
                # Extract Pareto frontier for this specific knob value
                x_pts, y_pts = val_sub[LAT_COL].values, val_sub[ycol].values
                pf_mask = pareto_frontier_2d(x_pts, y_pts)
                
                color = cmap(norm(v_idx))
                # Scatter points (background)
                ax.scatter(x_pts, y_pts, color=color, alpha=0.15, s=6, linewidths=0)
                
                # Step Line (Frontier)
                if pf_mask.sum() >= 2:
                    lx, ly = pareto_step_line(x_pts[pf_mask], y_pts[pf_mask])
                    ax.plot(lx, ly, color=color, linewidth=2, label=str(val), alpha=0.9, zorder=5)

            if j == cols - 1: # Add legend to the rightmost plot of each row
                ax.legend(title=klabel, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
def format_log_axis(ax, axis="x"):
    """Helper for clean log axis labels."""
    from matplotlib.ticker import LogLocator, FuncFormatter
    if "x" in axis:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    if "y" in axis:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))

def plot_architectural_sensitivities(tech, df, out_dir):
    """Simple sensitivity plots for WW, Assoc, Stacking."""
    knobs = ["word_width", "associativity", "stacked_die_count"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Architectural Metric Sensitivity — {tech}", fontsize=16)
    
    for ax, knob in zip(axes, knobs):
        if knob not in df.columns: continue
        df.groupby(knob)[LAT_COL].mean().plot(kind='bar', ax=ax, color='teal', alpha=0.7)
        ax.set_title(f"Impact of {knob}")
        ax.set_ylabel("Avg Latency (ns)")
        ax.grid(axis='y', alpha=0.3)
        
    path = os.path.join(out_dir, f"{tech}_sensitivities.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

def plot_capacity_scaling(tech, df, out_dir):
    """Plot PPA scaling across capacity points."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for metric, label, scale in METRICS:
        scaled = df.groupby(CAP_COL)[metric].mean()
        scaled.plot(ax=ax, label=label, marker='o')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Capacity (MB)")
    ax.set_ylabel("Normalized Metrics")
    ax.set_title(f"Technology Scaling Trend — {tech}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    path = os.path.join(out_dir, f"{tech}_scaling_trends.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

def main(tech: str):
    full_csv = f"pareto/{tech}_arch/{tech}_arch_full_data.csv"
    if not os.path.exists(full_csv):
        print(f"ERROR: {full_csv} not found. Run pareto_analysis.py --arch first.")
        return

    print(f"Loading architectural sweep data: {full_csv}")
    df = pd.read_csv(full_csv)
    
    out_dir = f"pareto/plots/{tech}_arch"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"INFO: Generating dashboards for {tech}_arch")
    plot_architectural_sensitivities(tech, df, out_dir)
    plot_capacity_scaling(tech, df, out_dir)
    plot_pareto_shift_matrix(tech, df, out_dir)
    
    print("\nVisualization Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tech", default="SRAM")
    args = p.parse_args()
    main(args.tech)

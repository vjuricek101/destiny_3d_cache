import os
import re
from typing import Dict, List, Tuple, Optional

# ── Column Selection ───────────────────────────────────────────────────────────
# Columns kept in the final CSV.  Dropped columns fall into 3 categories:
#   1. Array-level PPA (data_read_latency_ns, data_bank_area_mm2, etc.) —
#      these are direct components of the cache-level targets; including them
#      would let the model trivially "add up" sub-components (target leakage).
#   2. Tag-array duplicates (tag_*) — the data array organization already
#      captures the structural information; tag details add noise for training.
#   3. Dimension/efficiency columns (bank_height_um, area_efficiency_pct, etc.)
#      — these are outputs of the layout solver, not structural design decisions,
#      and are highly correlated with the PPA targets we predict.
KEEP_COLS: List[str] = [
    # ── Variant identifier (added by Python, not DESTINY) ──────────────────
    "variant_name",
    # ── Swept / configuration inputs ──────────────────────────────────────
    "opt_target",
    "capacity_kb",
    "word_width_bits",
    "associativity",
    "temperature_K",
    "process_node_nm",
    "device_roadmap",
    "internal_sensing",
    "mem_cell_type",
    # ── Cache-level PPA targets ────────────────────────────────────────────
    "cache_access_mode",
    "cache_area_mm2",
    "cache_hit_latency_ns",
    "cache_miss_latency_ns",
    "cache_write_latency_ns",
    "cache_refresh_latency_ns",
    "cache_hit_energy_nJ",
    "cache_miss_energy_nJ",
    "cache_write_energy_nJ",
    "cache_refresh_energy_nJ",
    "cache_leakage_mW",
    "cache_refresh_power_W",
    # ── Structural / organizational outputs (data array only) ──────────────
    # These describe HOW DESTINY organized the memory — safe model features.
    "data_stacked_die_count",    # destiny_bank_stacked
    "data_total_banks",          # destiny_total_banks
    "data_total_mats",           # destiny_total_mats
    "data_num_row_subarray",     # destiny_mat_rows
    "data_num_col_subarray",     # destiny_mat_cols
    "data_subarray_num_row",     # destiny_subarray_rows
    "data_subarray_num_col",     # destiny_subarray_cols
    "data_mux_sense_amp",        # destiny_senseamp_mux
    "data_mux_output_lev2",      # destiny_output_mux_l2
    "data_num_active_mat_per_col",
    "data_num_active_mat_per_row",
    "data_num_active_subarray_per_col",
    "data_num_active_subarray_per_row",
    "data_num_row_per_set",
]

# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_cell_params(filepath: str) -> Dict[str, str]:
    """Lightweight parser for DESTINY .cell files."""
    params: Dict[str, str] = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'): continue
            if line.startswith('-'):
                parts = line[1:].split(':', 1)
                if len(parts) == 2:
                    params[parts[0].strip()] = parts[1].strip()
    return params

def extract_process_node(cell_filename: str) -> Optional[int]:
    m = re.search(r'_n(\d+)\.cell$', cell_filename)
    return int(m.group(1)) if m else None

def setup_dirs(mem_type: str, is_arch: bool = False) -> Tuple[str, str]:
    suffix = "_arch" if is_arch else ""
    temp_dir    = f"/dev/shm/vjuricek_destiny_tmp/{mem_type}{suffix}"
    results_dir = f"exploration_results/{mem_type}{suffix}"
    for d in [temp_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    return temp_dir, results_dir

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

# Style
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
    ("cache_hit_latency_ns",   "Read Latency",  "log"),
    ("cache_write_energy_nJ",  "Write Energy",  "log"),
    ("cache_area_mm2",         "Area",          "log"),
    ("cache_leakage_mW",       "Leakage Power", "log"),
]
LAT_COL   = "cache_hit_latency_ns"
CAP_COL   = "capacity_kb"
TECH_COL  = "mem_cell_type"
TECHS     = ["SRAM", "RRAM", "eDRAM"]

# Marker shapes per technology — readable in grayscale/print
TECH_MARKERS = {"SRAM": "o", "RRAM": "s", "eDRAM": "^"}
TECH_COLORS  = {"SRAM": "#2196F3", "RRAM": "#E91E63", "eDRAM": "#4CAF50"}
SUB_MARKERS = {"CMOS": "o", "diode": "^", "none": "s", "HP": "o", "LOP": "^", "LSTP": "s", "EDRAM": "o"}

# Actual expected values from DESTINY sweep
CAP_KB_LABELS = {
    2: "2KB", 4: "4KB", 8: "8KB", 16: "16KB", 32: "32KB", 64: "64KB",
    128: "128KB", 256: "256KB", 512: "512KB", 1024: "1MB", 2048: "2MB",
    4096: "4MB", 8192: "8MB", 16384: "16MB", 32768: "32MB"
}

# Helpers

def pareto_frontier_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of non-dominated points (minimizing both x and y).
    A point is dominated if another point is ≤ in both dimensions and < in at least one.
    """
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        # Check whether any other point dominates i
        dominated[i] = np.any((x <= x[i]) & (y <= y[i]) &
                              ((x < x[i]) | (y < y[i])))
    return ~dominated

def pareto_step_line(x: np.ndarray, y: np.ndarray):
    """
    Sort Pareto-optimal points and return a stepped-line (x, y) path
    so the frontier can be drawn as a solid staircase instead of dots.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # Build staircase
    sx, sy = [xs[0]], [ys[0]]
    for xi, yi in zip(xs[1:], ys[1:]):
        sy.append(sy[-1])   # horizontal step
        sx.append(xi)
        sx.append(xi)       # vertical step
        sy.append(yi)
    return np.array(sx), np.array(sy)

def cap_colormap(caps: pd.Series):
    """Return a LogNorm and the sorted unique capacity values."""
    uniq = sorted(caps.unique())
    vmin, vmax = min(uniq), max(uniq)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    return norm, uniq

def add_cap_colorbar(fig, axes_list, norm, label="Capacity"):
    """Add a colorbar to the right of the rightmost panel without overlapping."""
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    ax_arg = axes_list if isinstance(axes_list, list) else [axes_list]

    cbar = fig.colorbar(sm, ax=ax_arg, pad=0.02, fraction=0.04)
    tick_vals = sorted(CAP_KB_LABELS.keys())
    cbar.set_ticks([v for v in tick_vals if norm.vmin <= v <= norm.vmax])
    cbar.set_ticklabels([CAP_KB_LABELS[v] for v in tick_vals
                         if norm.vmin <= v <= norm.vmax])
    cbar.set_label(label, fontsize=9)

def format_log_axis(ax, axis="both"):
    """Replace default 1e-X notation with plain decimal numbers."""
    fmt = ticker.LogFormatterSciNotation(labelOnlyBase=False)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, pos: f"{val:g}"))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, pos: f"{val:g}"))

def power2_xticks(ax, caps):
    """Set x-ticks at the exact power-of-2 capacity values present in the data."""
    tick_vals = sorted(caps)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([CAP_KB_LABELS.get(v, f"{v:.3g}") for v in tick_vals],
                       rotation=45, ha="right", fontsize=7.5)

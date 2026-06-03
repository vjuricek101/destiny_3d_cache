import math
import os
import re
from typing import Dict, Any, List, Tuple, Optional

# -- Centralized Target Definitions ---------------------------------------------
TARGET_COLS: List[str] = [
    "cache_area_mm2",
    "cache_hit_latency_ns",
    "cache_write_latency_ns",
    "cache_refresh_latency_ns",
    "cache_hit_energy_nJ",
    "cache_write_energy_nJ",
    "cache_refresh_energy_nJ",
    "cache_leakage_mW",
]

TARGET_LABELS: List[str] = [
    "Area (mm^2)",
    "Read Latency (ns)",
    "Write Latency (ns)",
    "Refresh Latency (ns)",
    "Read Energy (nJ)",
    "Write Energy (nJ)",
    "Refresh Energy (nJ)",
    "Leakage (mW)",
]

TARGET_SHORT_LABELS: List[str] = [
    "Area",
    "ReadLatency",
    "WriteLatency",
    "RefreshLatency",
    "ReadEnergy",
    "WriteEnergy",
    "RefreshEnergy",
    "Leakage",
]

TARGET_KEY_TO_OPT_TARGET: Dict[str, str] = {
    "cache_area_mm2":           "Area",
    "cache_hit_latency_ns":     "ReadLatency",
    "cache_write_latency_ns":    "WriteLatency",
    "cache_refresh_latency_ns":  "ReadLatency",
    "cache_hit_energy_nJ":       "ReadDynamicEnergy",
    "cache_write_energy_nJ":     "WriteDynamicEnergy",
    "cache_refresh_energy_nJ":   "ReadDynamicEnergy",
    "cache_leakage_mW":         "LeakagePower",
}

# Exclude targets that are zero for a given technology (e.g. SRAM refresh)
TECH_SKIP_TARGETS: Dict[str, List[str]] = {
    "SRAM":  ["cache_refresh_latency_ns", "cache_refresh_energy_nJ"],
    "RRAM":  [],
    "eDRAM": [],
}

def get_active_targets(tech: str) -> List[str]:
    """Return TARGET_COLS with tech-specific structural-zero columns removed."""
    skip = set(TECH_SKIP_TARGETS.get(tech, []))
    return [c for c in TARGET_COLS if c not in skip]


def get_active_labels(tech: str) -> List[str]:
    """TARGET_LABELS filtered to the same active subset as get_active_targets()."""
    skip = set(TECH_SKIP_TARGETS.get(tech, []))
    return [lbl for col, lbl in zip(TARGET_COLS, TARGET_LABELS) if col not in skip]


def get_active_short_labels(tech: str) -> List[str]:
    """TARGET_SHORT_LABELS filtered to the same active subset."""
    skip = set(TECH_SKIP_TARGETS.get(tech, []))
    return [lbl for col, lbl in zip(TARGET_COLS, TARGET_SHORT_LABELS) if col not in skip]


def build_fixed_context(
    node: int,
    roadmap: str,
    temperature: float,
    **fixed_params,
) -> Dict[str, Any]:
    """Build the ``fixed_context`` dict passed to every inverse-design optimizer.
    """
    ctx: Dict[str, Any] = {
        f"process_node_nm_{node}":   1.0,
        f"device_roadmap_{roadmap}": 1.0,
        "temperature_K":              temperature,
    }
    ctx.update({k: v for k, v in fixed_params.items() if v is not None})
    return ctx


# -- Inverse Design Constants & Vocabs ------------------------------
import torch

_MUX_VALID = [1, 2, 4, 8, 16, 32, 64]
_MAT_VALID = [1, 2, 4, 8, 16]

BASE_ARCH_COLS   = ["capacity_kb", "word_width_bits", "associativity", "data_stacked_die_count"]
BASE_ARCH_BOUNDS = [
    (math.log10(2),  math.log10(32768)),  # capacity_kb  (log10)
    (6.0,            11.0),               # word_width    (log2)
    (0.0,            6.0),                # associativity (log2)
    (0.0,            4.0),                # stacked_die   (log2)
]

DATA_PARAM_BOUNDS_LOG2 = {
    "data_mux_sense_amp":           (0.0, 6.0),
    "data_mux_output_lev1":         (0.0, 6.0),
    "data_mux_output_lev2":         (0.0, 6.0),
    "data_num_active_mat_per_row":  (0.0, 4.0),
    "data_num_active_mat_per_col":  (0.0, 4.0),
    "tag_num_row_mat":              (0.0, 6.0),
    "tag_num_col_mat":              (0.0, 6.0),
    "tag_mux_sense_amp":            (0.0, 6.0),
    "tag_mux_output_lev1":          (0.0, 6.0),
    "tag_mux_output_lev2":          (0.0, 6.0),
    "tag_num_active_mat_per_row":   (0.0, 6.0),
    "tag_num_active_mat_per_col":   (0.0, 6.0),
}
DATA_PARAM_BOUNDS_LINEAR = {
    "data_num_active_subarray_per_row": (1.0, 2.0),
    "data_num_active_subarray_per_col": (1.0, 2.0),
    "tag_num_active_subarray_per_row":  (1.0, 2.0),
    "tag_num_active_subarray_per_col":  (1.0, 2.0),
    "CellInput_CellAspectRatio":        (0.8, 2.5),
}
SRAM_CELL_BOUNDS_LOG10 = {
    "CellInput_SRAMCellNMOSWidth (F)": (2.2, 2.5),
    "CellInput_SRAMCellPMOSWidth (F)": (1.0, 1.1),
    "CellInput_AccessCMOSWidth (F)":   (1.1, 1.25),
}
SRAM_CELL_BOUNDS_LINEAR = {
    "CellInput_ReadVoltage (V)": (0.5, 1.2),
}

_CAP_KB_PHYS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
_WW_PHYS     = [64, 128, 256, 512, 1024, 2048]   # 2^6 … 2^11
_ASSOC_PHYS  = [1, 2, 4, 8, 16, 32, 64]          # 2^0 … 2^6
_DIE_PHYS    = [1, 2, 4, 8, 16]                  # 2^0 … 2^4

def _log10_vocab(phys_values):
    return torch.tensor([math.log10(v) for v in phys_values], dtype=torch.float32)

def _log2_vocab(phys_values):
    return torch.tensor([math.log2(v) for v in phys_values], dtype=torch.float32)

def _linear_grid(lo, hi, step):
    n = max(2, round((hi - lo) / step) + 1)
    vals = [lo + i * step for i in range(n)]
    vals[-1] = hi
    return torch.tensor(vals, dtype=torch.float32)

def build_gumbel_vocabs(opt_cols: list, tech: str) -> dict:
    vocabs = {}
    for col in opt_cols:
        if col == "capacity_kb":
            vocabs[col] = _log10_vocab(_CAP_KB_PHYS)
        elif col == "word_width_bits":
            vocabs[col] = _log2_vocab(_WW_PHYS)
        elif col == "associativity":
            vocabs[col] = _log2_vocab(_ASSOC_PHYS)
        elif col == "data_stacked_die_count":
            vocabs[col] = _log2_vocab(_DIE_PHYS)
        elif "mux_" in col:
            vocabs[col] = _log2_vocab(_MUX_VALID)
        elif "_mat_" in col or "num_row_mat" in col or "num_col_mat" in col:
            vocabs[col] = _log2_vocab(_MAT_VALID)
        elif "_subarray_" in col:
            vocabs[col] = torch.tensor([1.0, 2.0], dtype=torch.float32)
        elif tech == "SRAM" and col in SRAM_CELL_BOUNDS_LOG10:
            lo_phys, hi_phys = SRAM_CELL_BOUNDS_LOG10[col]
            vocabs[col] = _linear_grid(math.log10(lo_phys), math.log10(hi_phys), step=0.01)
        elif tech == "SRAM" and col in SRAM_CELL_BOUNDS_LINEAR:
            lo, hi = SRAM_CELL_BOUNDS_LINEAR[col]
            vocabs[col] = _linear_grid(lo, hi, step=0.05)
        else:
            vocabs[col] = torch.tensor([0.0, 1.0], dtype=torch.float32)
    return vocabs

def build_dyn_idx(feature_cols, opt_cols, opt_cats=None, cat_vocabs=None):
    dyn_keys = set(opt_cols) | {
        "derived_sqrt_capacity", "derived_cap_per_die", "derived_rows_per_die",
        "CellInput_CellArea (F^2)", "derived_sqrt_area", "derived_read_v_sq",
        "CellInput_MinSenseVoltage (mV)",
    } | {f"device_roadmap_{rm}_x_log10_cap" for rm in ["HP", "LOP", "LSTP"]}

    if opt_cats and cat_vocabs:
        for cat in opt_cats:
            for val in cat_vocabs[cat]:
                dyn_keys.add(f"{cat}_{val}")

    return {k: i for i, k in enumerate(feature_cols) if k in dyn_keys}

def nearest_valid(val: float, valid_set: list) -> float:
    return min(valid_set, key=lambda x: abs(x - val))



def parse_fixed_arg(s: str):
    """Parse a single ``KEY=VALUE`` string into a ``(key, value)`` pair.
    """
    import argparse as _ap
    k, sep, v = s.partition("=")
    if not sep:
        raise _ap.ArgumentTypeError(
            f"Expected KEY=VALUE (e.g. capacity_kb=64), got: {s!r}"
        )
    k = k.strip()
    for coerce in (int, float):
        try:
            return k, coerce(v)
        except ValueError:
            pass
    return k, v.strip()

METRIC_META: Dict[str, Dict[str, str]] = {
    "cache_area_mm2":         {"label": "Area (mm2)",          "unit": "mm2"},
    "cache_hit_latency_ns":   {"label": "Read Latency (ns)",  "unit": "ns"},
    "cache_write_latency_ns": {"label": "Write Latency (ns)",  "unit": "ns"},
    "cache_refresh_latency_ns": {"label": "Refresh Latency (ns)", "unit": "ns"},
    "cache_hit_energy_nJ":    {"label": "Hit Energy (nJ)",     "unit": "nJ"},
    "cache_write_energy_nJ":  {"label": "Write Energy (nJ)",   "unit": "nJ"},
    "cache_refresh_energy_nJ":  {"label": "Refresh Energy (nJ)",  "unit": "nJ"},
    "cache_leakage_mW":       {"label": "Leakage (mW)",        "unit": "mW"},
    "cache_miss_latency_ns":  {"label": "Miss Latency (ns)",   "unit": "ns"},
}

METRIC_TO_PPA_LABEL: Dict[str, str] = {
    "cache_area_mm2":           "Area",
    "cache_hit_latency_ns":     "ReadLatency",
    "cache_write_latency_ns":    "WriteLatency",
    "cache_refresh_latency_ns":  "RefreshLatency",
    "cache_hit_energy_nJ":       "ReadEnergy",
    "cache_write_energy_nJ":     "WriteEnergy",
    "cache_refresh_energy_nJ":   "RefreshEnergy",
    "cache_leakage_mW":         "Leakage",
}

LAYOUT_COLS: List[str] = [
    "data_mux_sense_amp",
    "data_mux_output_lev2",
    "data_num_active_mat_per_col",
    "data_num_active_mat_per_row",
    "data_num_active_subarray_per_col",
    "data_num_active_subarray_per_row",
]

# -- Column Selection -----------------------------------------------------------
# Columns kept in the final CSV.  Dropped columns fall into 3 categories:
#   1. Array-level PPA (data_read_latency_ns, data_bank_area_mm2, etc.) --
#      these are direct components of the cache-level targets; including them
#      would let the model trivially "add up" sub-components (target leakage).
#   2. Tag-array duplicates (tag_*) -- the data array organization already
#      captures the structural information; tag details add noise for training.
#   3. Dimension/efficiency columns (bank_height_um, area_efficiency_pct, etc.)
#      -- these are outputs of the layout solver, not structural design decisions,
#      and are highly correlated with the PPA targets we predict.
KEEP_COLS: List[str] = [
    # -- Variant identifier (added by Python, not DESTINY) ------------------
    "variant_name",
    # -- Swept / configuration inputs --------------------------------------
    "opt_target",
    "capacity_kb",
    "word_width_bits",
    "associativity",
    "temperature_K",
    "process_node_nm",
    "device_roadmap",
    "internal_sensing",
    "mem_cell_type",
    # -- Cache-level PPA targets --------------------------------------------
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
    # -- Structural / organizational outputs (data array only) --------------
    "data_stacked_die_count",    # destiny_bank_stacked
    "data_total_banks",          # destiny_total_banks
    "data_total_mats",           # destiny_total_mats
    "data_num_row_subarray",     # destiny_mat_rows
    "data_num_col_subarray",     # destiny_mat_cols
    "data_subarray_num_row",     # destiny_subarray_rows
    "data_subarray_num_col",     # destiny_subarray_cols
    "data_mux_sense_amp",        # destiny_senseamp_mux
    "data_mux_output_lev1",      # destiny_output_mux_l1
    "data_mux_output_lev2",      # destiny_output_mux_l2
    "data_num_active_mat_per_col",
    "data_num_active_mat_per_row",
    "data_num_active_subarray_per_col",
    "data_num_active_subarray_per_row",
    "data_num_row_per_set",
    # -- Wire configuration (logged but DESTINY auto-selects unless forced) ---
    "data_local_wire_type",
    "data_local_wire_repeater_type",
    "data_local_wire_low_swing",
    "data_global_wire_type",
    "data_global_wire_repeater_type",
    "data_global_wire_low_swing",
    # -- Buffer design optimization (BufferDesignOptimization) ----------------
    "data_area_optimization_level",
    # -- Tag array structural columns (source of per-opt_target variation) ----
    # cache PPA = data + tag; the data array is forced/identical per file,
    # so ALL inter-row differences in cache_area/leakage/latency/energy come
    # from these tag columns being optimized differently per opt_target.
    "tag_num_row_mat",
    "tag_num_col_mat",
    "tag_mux_sense_amp",
    "tag_mux_output_lev1",
    "tag_mux_output_lev2",
    "tag_num_active_mat_per_col",
    "tag_num_active_mat_per_row",
    "tag_num_active_subarray_per_col",
    "tag_num_active_subarray_per_row",
    "tag_num_row_subarray",
    "tag_num_col_subarray",
    "tag_subarray_num_row",
    "tag_subarray_num_col",
    "tag_area_optimization_level",
    "tag_local_wire_type",
    "tag_local_wire_repeater_type",
    "tag_local_wire_low_swing",
    "tag_global_wire_type",
    "tag_global_wire_repeater_type",
    "tag_global_wire_low_swing",
    # Tag PPA sub-components (directly additive into cache totals)
    "tag_bank_area_mm2",
    "tag_read_latency_ns",
    "tag_write_latency_ns",
    "tag_read_energy_pJ",
    "tag_write_energy_pJ",
    "tag_leakage_mW",
]

# -- Utilities -----------------------------------------------------------------

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

def setup_dirs(mem_type: str) -> Tuple[str, str]:
    temp_dir    = f"/dev/shm/vjuricek_destiny_tmp/{mem_type}"
    results_dir = f"exploration_results/{mem_type}"
    for d in [temp_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    return temp_dir, results_dir

# -- Plotting ------------------------------------------------------------------
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
    ("cache_hit_latency_ns",     "Read Latency (ns)",    "log"),
    ("cache_write_latency_ns",   "Write Latency (ns)",   "log"),
    ("cache_refresh_latency_ns", "Refresh Latency (ns)", "log"),
    ("cache_hit_energy_nJ",      "Read Energy (nJ)",     "log"),
    ("cache_write_energy_nJ",    "Write Energy (nJ)",    "log"),
    ("cache_refresh_energy_nJ",  "Refresh Energy (nJ)",  "log"),
    ("cache_leakage_mW",         "Leakage (mW)",         "log"),
    ("cache_area_mm2",           "Area (mm^2)",          "log"),
]
LAT_COL   = "cache_hit_latency_ns"
CAP_COL   = "capacity_kb"
TECH_COL  = "mem_cell_type"
TECHS     = ["SRAM", "RRAM", "eDRAM"]

# Marker shapes per technology -- readable in grayscale/print
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
    A point is dominated if another point is <= in both dimensions and < in at least one.
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

def pareto_frontier_nd(costs: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of non-dominated points for N dimensions.
    costs is a (N_points, N_dimensions) array. Minimizing all dimensions.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            # A point is dominated if another point is <= in all dims and < in at least one dim.
            dominated = np.all(costs[is_efficient] <= c, axis=1) & np.any(costs[is_efficient] < c, axis=1)
            is_efficient[is_efficient] = ~dominated
            is_efficient[i] = True  # Always keep self in this check
    return is_efficient

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

# -- SRAM Physics: layout-aware area (Base+Scaling) + Pelgrom mismatch sensing -

_A_VTH: Dict[int, float] = {65: 5.0, 45: 4.0, 32: 3.0, 22: 2.5}  # A_Vth [mV.um]

def derive_sram_physical_params(params: Dict[str, Any], node: int) -> Dict[str, Any]:
    """Compute CellArea (F^2) and MinSenseVoltage (mV) from transistor widths."""
    w_pd = float(params["SRAMCellNMOSWidth (F)"])
    w_pu = float(params["SRAMCellPMOSWidth (F)"])
    w_pg = float(params["AccessCMOSWidth (F)"])
    if w_pd <= 0 or w_pu <= 0 or w_pg <= 0:
        raise ValueError(f"Transistor widths must be >0; got {w_pd}, {w_pu}, {w_pg}")
    # Area: base (well isolation + contacts) + width term + height term
    area = 55.0 + 30.0 * max(w_pd, w_pg) + 20.0 * (w_pu + 0.5)
    params["CellArea (F^2)"] = round(max(40.0, min(200.0, area)), 4)
    # Sensing: 6sigma Pelgrom mismatch at SA input (W_SA = 2xW_PG, L = 1F)
    v_sense = 6.0 * _A_VTH.get(node, 3.0) / math.sqrt(2.0 * w_pg)
    params["MinSenseVoltage (mV)"] = round(max(5.0, min(80.0, v_sense)), 4)
    return params

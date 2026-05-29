#!/usr/bin/env python3
"""
validate_existing_results.py
----------------------------
Reads already completed optimization sweep CSVs in a directory, executes DESTINY
validation on the top-N designs and trend points, populates `destiny_x` and `destiny_y` 
so they can be plotted immediately by `plot_multi_sweep.py`.
"""

import os
import sys
import argparse
import re
import pandas as pd
import numpy as np
import glob

# Import helpers from the multi_sweep code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from inverse_design_multi_sweep import (
        _validate_and_capture,
        _layout_from_row,
        pct_err,
        SRAM_TEMPERATURE_MAP,
        METRIC_TO_PPA_LABEL,
        _validate_trend_designs
    )
except ImportError:
    # Fallback to manual definitions if import fails
    SRAM_TEMPERATURE_MAP = {1: 300, 2: 363, 4: 380}
    
    def pct_err(predicted, target):
        return float("nan") if target == 0 else (predicted - target) / abs(target) * 100.0

    def _layout_from_row(row):
        mapping = {
            "data_mux_sense_amp":              "mux_sa",
            "data_mux_output_lev2":            "mux_ol2",
            "data_num_active_mat_per_col":     "act_mat_col",
            "data_num_active_mat_per_row":     "act_mat_row",
            "data_num_active_subarray_per_col": "act_sub_col",
            "data_num_active_subarray_per_row": "act_sub_row",
        }
        out = {}
        for src, dst in mapping.items():
            val = row.get(src) if isinstance(row, dict) else (
                row[src] if src in row.index else None
            )
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                out[dst] = int(val)
        return out

    # Minimal validate and capture fallback
    import threading
    import subprocess
    import shutil
    import time

    def _generate_destiny_configs(cell_file, cfg_file, cap_kb, ww, assoc, stack, temp,
                                   wn, wp, wac, read_voltage, node, roadmap, opt_target, layout):
        from destiny_utils import derive_sram_physical_params
        p = {"SRAMCellNMOSWidth (F)": wn, "SRAMCellPMOSWidth (F)": wp, "AccessCMOSWidth (F)": wac}
        derive_sram_physical_params(p, node)
        cell_content = f"""-MemCellType: SRAM
-CellArea (F^2): {p["CellArea (F^2)"]:.4f}
-SRAMCellNMOSWidth (F): {wn:.4f}
-SRAMCellPMOSWidth (F): {wp:.4f}
-AccessCMOSWidth (F): {wac:.4f}
-AccessType: CMOS
-MinSenseVoltage (mV): {p["MinSenseVoltage (mV)"]:.4f}
-CellAspectRatio: 1.4600
-ReadVoltage (V): {read_voltage:.4f}
-Stitching: 16
"""
        cfg_content = f"""-OptimizationTarget: {opt_target}
-EnablePruning: Yes
-Capacity (KB): {cap_kb}
-WordWidth (bit): {ww}
-Associativity (for cache only): {assoc}
-StackedDieCount: {stack}
-Temperature (K): {temp}
-DeviceRoadmap: {roadmap}
-MemoryCellInputFile: {os.path.abspath(cell_file)}
-ProcessNode: {node}
"""
        if layout.get("mux_sa")    is not None: cfg_content += f"-ForceMuxSenseAmp: {int(layout['mux_sa'])}\n"
        if layout.get("mux_ol2")   is not None: cfg_content += f"-ForceMuxOutputLev2: {int(layout['mux_ol2'])}\n"
        if layout.get("act_mat_col") is not None and layout.get("act_mat_row") is not None:
            c, r = int(layout["act_mat_col"]), int(layout["act_mat_row"])
            cfg_content += f"-ForceBank (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"
        if layout.get("act_sub_col") is not None and layout.get("act_sub_row") is not None:
            c, r = int(layout["act_sub_col"]), int(layout["act_sub_row"])
            cfg_content += f"-ForceMat (Total AxB, Active CxD): {c}x{r}, {c}x{r}\n"

        with open(cell_file, "w") as f: f.write(cell_content)
        with open(cfg_file,  "w") as f: f.write(cfg_content)
        return cell_content, cfg_content

    def _run_destiny_process(cfg_file, timeout, verbose):
        output_lines = []
        def _stream(pipe):
            for line in iter(pipe.readline, ""):
                output_lines.append(line.strip())
        process = subprocess.Popen(
            ["./destiny", cfg_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1,
        )
        t = threading.Thread(target=_stream, args=(process.stdout,), daemon=True)
        t.start()
        t.join(timeout=timeout)
        if process.poll() is None:
            process.kill()
        return process, "\n".join(output_lines)

    def _parse_destiny_output(stdout_text, csv_file):
        res = {}
        m = re.search(r"Read Latency\s*=\s*([\d.]+)\s*(ps|ns|us)", stdout_text)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            res["cache_hit_latency_ns"] = val / 1000 if unit == "ps" else (val * 1000 if unit == "us" else val)
        m = re.search(r"(?:Total|Cache) Area\s*=\s*([\d.]+)\s*(mm\^2|um\^2)", stdout_text)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            res["cache_area_mm2"] = val / 1e6 if unit == "um^2" else val
        m = re.search(r"Write Dynamic Energy\s*=\s*([\d.]+)\s*(p|n|u)?J", stdout_text)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            res["cache_write_energy_nJ"] = val / 1000 if unit == "p" else (val * 1000 if unit == "u" else val)
        m = re.search(r"Leakage Power\s*=\s*([\d.]+)\s*(p|n|u|m)?W", stdout_text)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            if unit == "u":    val /= 1000
            elif unit == "n":  val /= 1e6
            elif unit == "p":  val /= 1e9
            elif not unit:     val *= 1000
            res["cache_leakage_mW"] = val
        return res if res else None

    def _validate_and_capture(cap_kb, ww, assoc, stack, temp, wn, wp, wac, read_voltage,
                               node, roadmap, opt_target, layout, timeout, verbose, run_tag):
        cell_file = f"multisweep_val_{run_tag}.cell"
        cfg_file  = f"multisweep_val_{run_tag}.cfg"
        csv_file  = f"multisweep_val_{run_tag}.csv"
        try:
            _generate_destiny_configs(
                cell_file, cfg_file, cap_kb, ww, assoc, stack, temp,
                wn, wp, wac, read_voltage, node, roadmap, opt_target, layout
            )
            process, stdout_text = _run_destiny_process(cfg_file, timeout, verbose)
            return _parse_destiny_output(stdout_text, csv_file)
        except Exception:
            return None
        finally:
            for f in [cell_file, cfg_file, csv_file]:
                if os.path.exists(f): os.remove(f)

def validate_file(fpath, validate_top, n_trend, timeout, verbose):
    print(f"\n========================================================")
    print(f"Validating file: {os.path.basename(fpath)}")
    print(f"========================================================")
    
    df = pd.read_csv(fpath)
    if len(df) == 0:
        print("  x File is empty. Skipping.")
        return

    # 1. Parse metadata from the CSV columns & first row
    first = df.iloc[0]
    
    if "metrics_optimized" in df.columns:
        metrics_list = str(first["metrics_optimized"]).split()
        x_metric = metrics_list[0]
        y_metric = metrics_list[1] if len(metrics_list) > 1 else x_metric
    elif "x_metric" in df.columns and "y_metric" in df.columns:
        x_metric = first["x_metric"]
        y_metric = first["y_metric"]
    else:
        # Fallback parse from filename
        filename = os.path.basename(fpath)
        m_m = re.search(r"nm_[A-Z0-9]+_[a-z]+_(.*)_feas[0-9]\.csv", filename)
        if m_m:
            pair_str = m_m.group(1)
            pair = pair_str.split("_vs_")
            x_metric = pair[0]
            y_metric = pair[1] if len(pair) > 1 else pair[0]
        else:
            x_metric = "cache_hit_latency_ns"
            y_metric = "cache_area_mm2"
            
    variant = first.get("variant", "baseline")
    feasibility = bool(first.get("feasibility", False))
    node = int(first["node_nm"])
    
    # Parse roadmap from filename
    # File name structure: sweep_{tech}_{node}nm_{roadmap}_{variant}_{x_metric}_vs_{y_metric}_feas{0|1}.csv
    filename = os.path.basename(fpath)
    roadmap = "HP"
    m_rm = re.search(r"nm_([A-Z0-9]+)_", filename)
    if m_rm:
        roadmap = m_rm.group(1)
        
    tech = "SRAM"
    m_tech = re.search(r"sweep_([A-Za-z0-9]+)_", filename)
    if m_tech:
        tech = m_tech.group(1)

    print(f"  Metadata: Tech={tech} | Node={node}nm | Roadmap={roadmap} | Variant={variant}")
    print(f"  Metrics : X={x_metric} | Y={y_metric} | Feasibility={feasibility}")

    if "metrics_optimized" in df.columns:
        metrics = str(first["metrics_optimized"]).split()
    else:
        metrics = [x_metric, y_metric]

    # Map the columns to x/y so sorting and plotting works smoothly
    if f"target_{x_metric}" in df.columns:
        df["target_x"] = df[f"target_{x_metric}"]
    if f"target_{y_metric}" in df.columns:
        df["target_y"] = df[f"target_{y_metric}"]
    if f"surr_{x_metric}" in df.columns:
        df["surr_x"] = df[f"surr_{x_metric}"]
    if f"surr_{y_metric}" in df.columns:
        df["surr_y"] = df[f"surr_{y_metric}"]
    if f"post_snap_surr_{x_metric}" in df.columns:
        df["post_snap_surr_x"] = df[f"post_snap_surr_{x_metric}"]
    if f"post_snap_surr_{y_metric}" in df.columns:
        df["post_snap_surr_y"] = df[f"post_snap_surr_{y_metric}"]

    # Initialize plotting columns
    for col in ["destiny_x", "destiny_y", "destiny_mean_abs_err_pct"]:
        if col not in df.columns:
            df[col] = None
    for k in metrics:
        if f"destiny_{k}" not in df.columns:
            df[f"destiny_{k}"] = None
        if f"destiny_err_{k}_pct" not in df.columns:
            df[f"destiny_err_{k}_pct"] = None

    # Sort to rank by post-snap error
    ranked_idx = df["post_snap_surr_mean_abs_err_pct"].sort_values().head(validate_top).index
    
    print(f"\n--- Running DESTINY for Top-{validate_top} Absolute Point Accuracy ---")
    for rank, ri in enumerate(ranked_idx, 1):
        row = df.loc[ri]
        
        # Recover physical parameters
        wn  = float(row.get("opt_wn",  row.get("orig_wn",  2.5)))
        wp  = float(row.get("opt_wp",  row.get("orig_wp",  2.0)))
        wac = float(row.get("opt_wac", row.get("orig_wac", 2.5)))
        rv  = float(row.get("opt_rv",  row.get("orig_rv",  1.0)))
        stack = max(1, int(row.get("data_stacked_die_count", 1)))
        temp  = SRAM_TEMPERATURE_MAP.get(stack, 300)

        cap_kb   = int(row["capacity_kb"])
        ww       = int(row["word_width_bits"])
        assoc    = int(row["associativity"])
        layout   = _layout_from_row(row)
        run_tag  = f"exval_{rank}_{int(time.time())}"

        print(f"  [{rank}/{validate_top}] Cap={cap_kb}KB  Node={node}nm  post_snap_err={row['post_snap_surr_mean_abs_err_pct']:.1f}%")

        destiny_ppa = _validate_and_capture(
            cap_kb=cap_kb, ww=ww, assoc=assoc, stack=stack, temp=temp,
            wn=wn, wp=wp, wac=wac, read_voltage=rv,
            node=node, roadmap=roadmap, opt_target="ReadLatency",
            layout=layout, timeout=timeout, verbose=verbose, run_tag=run_tag
        )

        if destiny_ppa is None:
            print("    x DESTINY validation failed for this configuration.")
            continue

        d_errs = []
        out_strs = []
        
        # Map values to metric columns
        dx = destiny_ppa.get(x_metric)
        dy = destiny_ppa.get(y_metric)
        
        df.at[ri, "destiny_x"] = dx
        df.at[ri, "destiny_y"] = dy
        
        for idx, k in enumerate(metrics):
            dk = destiny_ppa.get(k)
            df.at[ri, f"destiny_{k}"] = dk
            if dk is not None:
                t_col = f"target_{k}" if f"target_{k}" in row.index else ("target_x" if idx == 0 else "target_y")
                err = pct_err(dk, row[t_col])
                df.at[ri, f"destiny_err_{k}_pct"] = err
                d_errs.append(abs(err))
                out_strs.append(f"{k}={dk:.4g} (err={err:+.1f}%)")
        
        d_mean = float(np.mean(d_errs)) if d_errs else float("nan")
        df.at[ri, "destiny_mean_abs_err_pct"] = d_mean
        print(f"    ✓ DESTINY → " + "  ".join(out_strs) + f"  mean={d_mean:.1f}%")

    # 2. Trend validation (n-trend-points)
    if n_trend > 0 and len(metrics) == 2:
        print(f"\n--- Running DESTINY for {n_trend} Trend Points ---")
        for col in ["trend_destiny_x", "trend_destiny_y"]:
            if col not in df.columns:
                df[col] = None

        df_sorted = df.sort_values("target_x").reset_index(drop=False)
        indices = np.linspace(0, len(df_sorted) - 1, min(n_trend, len(df_sorted)), dtype=int)
        trend_df = df_sorted.iloc[indices]
        
        for rank, (ri, row_tuple) in enumerate(zip(trend_df["index"], trend_df.itertuples()), 1):
            row = df.loc[ri]
            
            wn  = float(row.get("opt_wn",  row.get("orig_wn",  2.5)))
            wp  = float(row.get("opt_wp",  row.get("orig_wp",  2.0)))
            wac = float(row.get("opt_wac", row.get("orig_wac", 2.5)))
            rv  = float(row.get("opt_rv",  row.get("orig_rv",  1.0)))
            stack = max(1, int(row.get("data_stacked_die_count", 1)))
            temp  = SRAM_TEMPERATURE_MAP.get(stack, 300)

            cap_kb   = int(row["capacity_kb"])
            ww       = int(row["word_width_bits"])
            assoc    = int(row["associativity"])
            layout   = _layout_from_row(row)
            run_tag  = f"extrend_{rank}_{int(time.time())}"

            print(f"  [{rank}/{n_trend}] TargetX={row['target_x']:.4g} | TargetY={row['target_y']:.4g}")

            destiny_ppa = _validate_and_capture(
                cap_kb=cap_kb, ww=ww, assoc=assoc, stack=stack, temp=temp,
                wn=wn, wp=wp, wac=wac, read_voltage=rv,
                node=node, roadmap=roadmap, opt_target="ReadLatency",
                layout=layout, timeout=timeout, verbose=verbose, run_tag=run_tag
            )

            if destiny_ppa is None:
                print("    x DESTINY failed.")
                continue

            df.at[ri, "trend_destiny_x"] = destiny_ppa.get(x_metric)
            df.at[ri, "trend_destiny_y"] = destiny_ppa.get(y_metric)
            print(f"    ✓ DESTINY → x={destiny_ppa.get(x_metric):.4g}  y={destiny_ppa.get(y_metric):.4g}")

        # Compute correlations
        valid_trend = df.dropna(subset=["trend_destiny_x", "trend_destiny_y"])
        if len(valid_trend) > 1:
            from scipy.stats import kendalltau, spearmanr
            tau_x, _ = kendalltau(valid_trend["target_x"], valid_trend["trend_destiny_x"])
            tau_y, _ = kendalltau(valid_trend["target_y"], valid_trend["trend_destiny_y"])
            spr_x, _ = spearmanr(valid_trend["target_x"], valid_trend["trend_destiny_x"])
            spr_y, _ = spearmanr(valid_trend["target_y"], valid_trend["trend_destiny_y"])
            
            print(f"\n  ✓ Trend Correlations (n={len(valid_trend)}):")
            print(f"    X-axis ({x_metric}): Kendall Tau = {tau_x:.3f}, Spearman = {spr_x:.3f}")
            print(f"    Y-axis ({y_metric}): Kendall Tau = {tau_y:.3f}, Spearman = {spr_y:.3f}")
            
            # Save correlation row to global CSV
            corr_file = f"sweep_correlations_{tech}_{node}nm_{roadmap}.csv"
            corr_path = os.path.join(os.path.dirname(fpath), corr_file)
            row_dict = {
                "variant": variant,
                "feasibility": feasibility,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "n_points": len(valid_trend),
                "kendall_tau_x": tau_x,
                "spearman_x": spr_x,
                "kendall_tau_y": tau_y,
                "spearman_y": spr_y
            }
            file_exists = os.path.exists(corr_path)
            with open(corr_path, "a") as f:
                if not file_exists:
                    f.write(",".join(row_dict.keys()) + "\n")
                f.write(",".join(str(row_dict[k]) for k in row_dict.keys()) + "\n")
    elif n_trend > 0:
        print(f"\n--- Skipping Trend Validation (Only supported for 2D sweeps, file has {len(metrics)} metrics) ---")

    # Save CSV back
    df.to_csv(fpath, index=False)
    print(f"\n✓ Successfully updated CSV: {fpath}")

def main():
    p = argparse.ArgumentParser(description="Validate already completed optimization CSVs using DESTINY.")
    p.add_argument("--dir", default="benchmark_results", help="Directory containing sweep_*.csv files")
    p.add_argument("--validate-top", type=int, default=5, help="Number of top designs to validate (default: 5)")
    p.add_argument("--n-trend-points", type=int, default=5, help="Number of trend points to validate (default: 5)")
    p.add_argument("--timeout", type=int, default=30, help="DESTINY timeout in seconds")
    p.add_argument("--verbose", action="store_true", help="Print verbose DESTINY outputs")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.dir, "sweep_*.csv")) + glob.glob(os.path.join(args.dir, "benchmark_*.csv"))
    if not files:
        print(f"No sweep_*.csv or benchmark_*.csv files found in directory: {args.dir}")
        sys.exit(1)

    print(f"Found {len(files)} sweep/benchmark files to validate in '{args.dir}'.")
    for f in sorted(files):
        validate_file(f, args.validate_top, args.n_trend_points, args.timeout, args.verbose)

if __name__ == "__main__":
    import time
    main()

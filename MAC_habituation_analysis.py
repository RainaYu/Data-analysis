#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAC_habituation_analysis.py

Batch analysis for zebrafish habituation tracking data.

Main features:
1. Drop first 150 frames from every session
2. Remove obvious single-frame jump-based false detections
3. Quantify activity state
4. Quantify spatial exploration
5. Quantify whether behaviour becomes stable over time

Expected folder structure:
    /Volumes/Raina/habituation/
        2026_04_01_10_00_00/
            tracking.csv
        2026_04_01_14_00_00/
            tracking.csv
        ...

Outputs:
    /Volumes/Raina/habituation/habituation_analysis_outputs/
        session_summary.csv
        lane_summary.csv
        per_minute_metrics.csv
        cleaning_summary.csv
        figures/
            <session_name>_activity.png
            <session_name>_space.png
            <session_name>_stability.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# User settings
# =========================================================
ROOT_DIR = Path("/Volumes/Raina/habituation")
OUTPUT_DIR = ROOT_DIR / "habituation_analysis_outputs"
FIG_DIR = OUTPUT_DIR / "figures"

TRACKING_FILENAME = "tracking.csv"

ACTIVE_LANES = [2, 3, 4, 5, 6, 7, 8, 9]

MAX_DURATION_SEC = 30 * 60
BIN_SEC = 60
DROP_FIRST_FRAMES = 150

SPEED_STOP_THRESHOLD = 5.0  # px/s

LEFT_BOUND = 1 / 3
RIGHT_BOUND = 2 / 3

LOW_Q = 0.01
HIGH_Q = 0.99

X_BINS = 6
Y_BINS = 4

ROLLING_MINUTES = 3

# =========================================================
# False detection cleaning settings
# =========================================================
STABLE_WINDOW = 5
STABLE_STEP_MAX_PX = 20.0
JUMP_MULTIPLIER = 6.0
ABS_JUMP_MIN_PX = 60.0
RETURN_TOLERANCE_PX = 25.0


# =========================================================
# Utility
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_div(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def shannon_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return np.nan
    return -np.sum(probs * np.log2(probs))


def fit_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    coef = np.polyfit(x, y, 1)
    return coef[0]


# =========================================================
# Load
# =========================================================
def load_tracking_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "relative_time_sec",
        "frame_id",
        "lane",
        "detected",
        "cx",
        "cy",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.copy()
    df["session"] = csv_path.parent.name

    for c in ["relative_time_sec", "frame_id", "lane", "detected", "cx", "cy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    optional_cols = ["fps", "area", "conf", "x1", "y1", "x2", "y2"]
    for c in optional_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["lane"].isin(ACTIVE_LANES)].copy()
    df = df[df["relative_time_sec"] <= MAX_DURATION_SEC].copy()

    df = df[df["frame_id"] > DROP_FIRST_FRAMES].copy()

    min_time = df["relative_time_sec"].min()
    if pd.notna(min_time):
        df["relative_time_sec"] = df["relative_time_sec"] - min_time

    return df


# =========================================================
# Robust lane normalization
# =========================================================
def normalize_lane_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust within-lane normalization.
    Even if some lanes have too few valid detections, do not crash.
    """
    out = df.copy()

    out["cx_low"] = np.nan
    out["cx_high"] = np.nan
    out["cy_low"] = np.nan
    out["cy_high"] = np.nan
    out["x_norm"] = np.nan
    out["y_norm"] = np.nan

    valid = out[
        (out["detected"] == 1)
        & (out["cx"].notna())
        & (out["cy"].notna())
        & (out["lane"].notna())
    ].copy()

    if valid.empty:
        return out

    lane_rows = []

    for lane, g in valid.groupby("lane"):
        if len(g) < 5:
            continue

        cx_low = g["cx"].quantile(LOW_Q)
        cx_high = g["cx"].quantile(HIGH_Q)
        cy_low = g["cy"].quantile(LOW_Q)
        cy_high = g["cy"].quantile(HIGH_Q)

        if pd.isna(cx_low) or pd.isna(cx_high) or cx_high <= cx_low:
            continue
        if pd.isna(cy_low) or pd.isna(cy_high) or cy_high <= cy_low:
            continue

        lane_rows.append({
            "lane": lane,
            "cx_low": cx_low,
            "cx_high": cx_high,
            "cy_low": cy_low,
            "cy_high": cy_high,
        })

    if not lane_rows:
        return out

    lane_stats = pd.DataFrame(lane_rows)

    out = out.drop(columns=["cx_low", "cx_high", "cy_low", "cy_high"], errors="ignore")
    out = out.merge(lane_stats, on="lane", how="left")

    out["x_norm"] = (out["cx"] - out["cx_low"]) / (out["cx_high"] - out["cx_low"])
    out["y_norm"] = (out["cy"] - out["cy_low"]) / (out["cy_high"] - out["cy_low"])

    out["x_norm"] = out["x_norm"].clip(0, 1)
    out["y_norm"] = out["y_norm"].clip(0, 1)

    return out


# =========================================================
# Framewise speed
# =========================================================
def compute_framewise_speed(df: pd.DataFrame) -> pd.DataFrame:
    out_list = []

    for lane, g in df.groupby("lane"):
        g = g.sort_values("relative_time_sec").copy()

        g["prev_t"] = g["relative_time_sec"].shift(1)
        g["prev_x"] = g["cx"].shift(1)
        g["prev_y"] = g["cy"].shift(1)
        g["prev_detected"] = g["detected"].shift(1)

        g["dt"] = g["relative_time_sec"] - g["prev_t"]
        g["dx"] = g["cx"] - g["prev_x"]
        g["dy"] = g["cy"] - g["prev_y"]
        g["step_distance_px"] = np.sqrt(g["dx"] ** 2 + g["dy"] ** 2)
        g["speed_px_s"] = g["step_distance_px"] / g["dt"]

        invalid = (
            (g["detected"] != 1)
            | (g["prev_detected"] != 1)
            | (~np.isfinite(g["dt"]))
            | (g["dt"] <= 0)
        )
        g.loc[invalid, ["step_distance_px", "speed_px_s"]] = np.nan

        out_list.append(g)

    if not out_list:
        return df.copy()

    return pd.concat(out_list, ignore_index=True)


# =========================================================
# False detection cleaning
# =========================================================
def detect_single_frame_spikes(g: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative rule:
    If previous steps are stable, current frame jumps far away,
    and next frame returns near the previous position,
    then current frame is likely a false detection spike.
    """
    g = g.sort_values("relative_time_sec").copy()
    g["false_jump_spike"] = False

    valid_idx = g.index[g["detected"] == 1].tolist()
    if len(valid_idx) < STABLE_WINDOW + 2:
        return g

    valid = g.loc[valid_idx, ["relative_time_sec", "cx", "cy"]].copy()
    valid["prev_x"] = valid["cx"].shift(1)
    valid["prev_y"] = valid["cy"].shift(1)
    valid["next_x"] = valid["cx"].shift(-1)
    valid["next_y"] = valid["cy"].shift(-1)

    valid["step_in"] = np.sqrt((valid["cx"] - valid["prev_x"]) ** 2 + (valid["cy"] - valid["prev_y"]) ** 2)
    valid["step_out"] = np.sqrt((valid["next_x"] - valid["cx"]) ** 2 + (valid["next_y"] - valid["cy"]) ** 2)
    valid["return_gap"] = np.sqrt((valid["next_x"] - valid["prev_x"]) ** 2 + (valid["next_y"] - valid["prev_y"]) ** 2)

    valid["baseline_prev_step"] = valid["step_in"].rolling(
        window=STABLE_WINDOW,
        min_periods=STABLE_WINDOW
    ).median().shift(1)

    valid["stable_prev"] = valid["step_in"].rolling(
        window=STABLE_WINDOW,
        min_periods=STABLE_WINDOW
    ).max().shift(1) < STABLE_STEP_MAX_PX

    valid["jump_threshold"] = np.maximum(
        ABS_JUMP_MIN_PX,
        valid["baseline_prev_step"] * JUMP_MULTIPLIER
    )

    suspicious = (
        valid["stable_prev"].fillna(False)
        & (valid["step_in"] > valid["jump_threshold"])
        & (valid["step_out"] > valid["jump_threshold"])
        & (valid["return_gap"] < RETURN_TOLERANCE_PX)
    )

    suspicious_idx = valid.index[suspicious].tolist()
    g.loc[suspicious_idx, "false_jump_spike"] = True

    return g


def remove_false_detections(df: pd.DataFrame):
    cleaned_list = []
    cleaning_rows = []

    for lane, g in df.groupby("lane"):
        g = detect_single_frame_spikes(g)

        before_detected = int((g["detected"] == 1).sum())
        removed_spikes = int(g["false_jump_spike"].sum())

        spike_mask = g["false_jump_spike"] & (g["detected"] == 1)
        g.loc[spike_mask, "detected"] = 0
        g.loc[spike_mask, ["cx", "cy"]] = np.nan

        if "area" in g.columns:
            g.loc[spike_mask, "area"] = np.nan
        if "conf" in g.columns:
            g.loc[spike_mask, "conf"] = np.nan
        if all(col in g.columns for col in ["x1", "y1", "x2", "y2"]):
            g.loc[spike_mask, ["x1", "y1", "x2", "y2"]] = np.nan

        after_detected = int((g["detected"] == 1).sum())

        cleaned_list.append(g)
        cleaning_rows.append({
            "session": g["session"].iloc[0],
            "lane": lane,
            "detected_before_cleaning": before_detected,
            "removed_false_jump_spikes": removed_spikes,
            "detected_after_cleaning": after_detected,
            "removed_ratio_within_detected": safe_div(removed_spikes, before_detected),
        })

    cleaned = pd.concat(cleaned_list, ignore_index=True) if cleaned_list else df.copy()
    cleaning_summary = pd.DataFrame(cleaning_rows)

    return cleaned, cleaning_summary


# =========================================================
# Space features
# =========================================================
def assign_x_zone(x_norm):
    if pd.isna(x_norm):
        return np.nan
    if x_norm < LEFT_BOUND:
        return "left"
    elif x_norm < RIGHT_BOUND:
        return "middle"
    return "right"


def add_space_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["x_zone"] = out["x_norm"].apply(assign_x_zone)

    out["x_bin"] = np.floor(out["x_norm"] * X_BINS).clip(0, X_BINS - 1)
    out["y_bin"] = np.floor(out["y_norm"] * Y_BINS).clip(0, Y_BINS - 1)

    out["space_bin"] = (
        out["x_bin"].astype("Int64").astype(str)
        + "_"
        + out["y_bin"].astype("Int64").astype(str)
    )

    out.loc[out["detected"] != 1, ["x_zone", "space_bin"]] = np.nan
    return out


# =========================================================
# Summaries
# =========================================================
def summarize_lane_session(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    rows = []

    for lane, g in df.groupby("lane"):
        g = g.sort_values("relative_time_sec").copy()
        valid = g[g["detected"] == 1].copy()

        total_rows = len(g)
        detected_rows = len(valid)
        detection_rate = safe_div(detected_rows, total_rows)

        mean_speed = valid["speed_px_s"].mean()
        median_speed = valid["speed_px_s"].median()
        p90_speed = valid["speed_px_s"].quantile(0.90)
        total_distance = valid["step_distance_px"].sum()

        stop_ratio = np.mean(valid["speed_px_s"] < SPEED_STOP_THRESHOLD) if len(valid) > 0 else np.nan

        zone_counts = valid["x_zone"].value_counts(normalize=True)
        left_ratio = zone_counts.get("left", 0.0)
        middle_ratio = zone_counts.get("middle", 0.0)
        right_ratio = zone_counts.get("right", 0.0)

        zone_entropy = shannon_entropy(np.array([left_ratio, middle_ratio, right_ratio]))

        valid["prev_zone"] = valid["x_zone"].shift(1)
        transitions = np.sum(
            (valid["x_zone"].notna())
            & (valid["prev_zone"].notna())
            & (valid["x_zone"] != valid["prev_zone"])
        )
        transition_rate = safe_div(transitions, MAX_DURATION_SEC / 60.0)

        unique_bins = valid["space_bin"].nunique()
        total_bins = X_BINS * Y_BINS
        spatial_coverage = safe_div(unique_bins, total_bins)

        rows.append({
            "session": session_name,
            "lane": lane,
            "detection_rate": detection_rate,
            "mean_speed_px_s": mean_speed,
            "median_speed_px_s": median_speed,
            "p90_speed_px_s": p90_speed,
            "total_distance_px": total_distance,
            "stop_ratio": stop_ratio,
            "left_ratio": left_ratio,
            "middle_ratio": middle_ratio,
            "right_ratio": right_ratio,
            "zone_entropy_bits": zone_entropy,
            "zone_transitions_per_min": transition_rate,
            "spatial_coverage": spatial_coverage,
        })

    return pd.DataFrame(rows)


def compute_per_minute_metrics(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    rows = []

    for lane, g in df.groupby("lane"):
        g = g.sort_values("relative_time_sec").copy()

        for minute_start in range(0, MAX_DURATION_SEC, BIN_SEC):
            minute_end = minute_start + BIN_SEC
            w = g[(g["relative_time_sec"] >= minute_start) & (g["relative_time_sec"] < minute_end)].copy()
            valid = w[w["detected"] == 1].copy()

            detection_rate = safe_div(len(valid), len(w))
            mean_speed = valid["speed_px_s"].mean()
            total_distance = valid["step_distance_px"].sum()
            stop_ratio = np.mean(valid["speed_px_s"] < SPEED_STOP_THRESHOLD) if len(valid) > 0 else np.nan

            zone_counts = valid["x_zone"].value_counts(normalize=True)
            left_ratio = zone_counts.get("left", 0.0)
            middle_ratio = zone_counts.get("middle", 0.0)
            right_ratio = zone_counts.get("right", 0.0)
            zone_entropy = shannon_entropy(np.array([left_ratio, middle_ratio, right_ratio]))

            valid["prev_zone"] = valid["x_zone"].shift(1)
            transitions = np.sum(
                (valid["x_zone"].notna())
                & (valid["prev_zone"].notna())
                & (valid["x_zone"] != valid["prev_zone"])
            )

            unique_bins = valid["space_bin"].nunique()
            spatial_coverage = safe_div(unique_bins, X_BINS * Y_BINS)

            rows.append({
                "session": session_name,
                "lane": lane,
                "minute_idx": minute_start // 60,
                "minute_start_sec": minute_start,
                "minute_end_sec": minute_end,
                "detection_rate": detection_rate,
                "mean_speed_px_s": mean_speed,
                "total_distance_px": total_distance,
                "stop_ratio": stop_ratio,
                "left_ratio": left_ratio,
                "middle_ratio": middle_ratio,
                "right_ratio": right_ratio,
                "zone_entropy_bits": zone_entropy,
                "zone_transitions": transitions,
                "spatial_coverage": spatial_coverage,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    avg_rows = (
        out.groupby(["session", "minute_idx", "minute_start_sec", "minute_end_sec"], as_index=False)
        .agg(
            lane=("lane", lambda s: "mean_active_lanes"),
            detection_rate=("detection_rate", "mean"),
            mean_speed_px_s=("mean_speed_px_s", "mean"),
            total_distance_px=("total_distance_px", "mean"),
            stop_ratio=("stop_ratio", "mean"),
            left_ratio=("left_ratio", "mean"),
            middle_ratio=("middle_ratio", "mean"),
            right_ratio=("right_ratio", "mean"),
            zone_entropy_bits=("zone_entropy_bits", "mean"),
            zone_transitions=("zone_transitions", "mean"),
            spatial_coverage=("spatial_coverage", "mean"),
        )
    )

    out = pd.concat([out, avg_rows], ignore_index=True)
    return out


def compute_stability_summary(per_minute: pd.DataFrame, session_name: str) -> dict:
    g = per_minute[
        (per_minute["session"] == session_name)
        & (per_minute["lane"] == "mean_active_lanes")
    ].sort_values("minute_idx").copy()

    if g.empty:
        return {
            "session": session_name,
            "early_mean_speed_px_s": np.nan,
            "middle_mean_speed_px_s": np.nan,
            "late_mean_speed_px_s": np.nan,
            "speed_slope_per_min": np.nan,
            "late_speed_cv": np.nan,
            "zone_entropy_slope_per_min": np.nan,
            "late_zone_bias_change": np.nan,
            "behaviour_stabilizing": np.nan,
        }

    early = g[g["minute_idx"].between(0, 9)]
    middle = g[g["minute_idx"].between(10, 19)]
    late = g[g["minute_idx"].between(20, 29)]

    early_mean_speed = early["mean_speed_px_s"].mean()
    middle_mean_speed = middle["mean_speed_px_s"].mean()
    late_mean_speed = late["mean_speed_px_s"].mean()

    speed_slope = fit_slope(g["minute_idx"], g["mean_speed_px_s"])
    entropy_slope = fit_slope(g["minute_idx"], g["zone_entropy_bits"])

    late_mean = late["mean_speed_px_s"].mean()
    late_speed_cv = late["mean_speed_px_s"].std() / late_mean if pd.notna(late_mean) and late_mean != 0 else np.nan

    middle_zone = np.array([
        middle["left_ratio"].mean(),
        middle["middle_ratio"].mean(),
        middle["right_ratio"].mean(),
    ], dtype=float)

    late_zone = np.array([
        late["left_ratio"].mean(),
        late["middle_ratio"].mean(),
        late["right_ratio"].mean(),
    ], dtype=float)

    late_zone_bias_change = np.nansum(np.abs(middle_zone - late_zone))

    behaviour_stabilizing = (
        (pd.notna(early_mean_speed) and pd.notna(late_mean_speed) and late_mean_speed < early_mean_speed)
        and (pd.notna(late_speed_cv) and late_speed_cv < 0.35)
    )

    return {
        "session": session_name,
        "early_mean_speed_px_s": early_mean_speed,
        "middle_mean_speed_px_s": middle_mean_speed,
        "late_mean_speed_px_s": late_mean_speed,
        "speed_slope_per_min": speed_slope,
        "late_speed_cv": late_speed_cv,
        "zone_entropy_slope_per_min": entropy_slope,
        "late_zone_bias_change": late_zone_bias_change,
        "behaviour_stabilizing": behaviour_stabilizing,
    }


def summarize_session(lane_summary: pd.DataFrame, stability_dict: dict, cleaning_summary: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "detection_rate",
        "mean_speed_px_s",
        "median_speed_px_s",
        "p90_speed_px_s",
        "total_distance_px",
        "stop_ratio",
        "left_ratio",
        "middle_ratio",
        "right_ratio",
        "zone_entropy_bits",
        "zone_transitions_per_min",
        "spatial_coverage",
    ]

    session_name = lane_summary["session"].iloc[0]
    out = lane_summary[numeric_cols].mean().to_dict()
    out["session"] = session_name

    out["removed_false_jump_spikes_total"] = cleaning_summary["removed_false_jump_spikes"].sum() if not cleaning_summary.empty else 0
    out["removed_false_jump_spikes_mean_per_lane"] = cleaning_summary["removed_false_jump_spikes"].mean() if not cleaning_summary.empty else 0
    out["removed_ratio_mean_per_lane"] = cleaning_summary["removed_ratio_within_detected"].mean() if not cleaning_summary.empty else 0

    out.update(stability_dict)

    return pd.DataFrame([out])


# =========================================================
# Plotting
# =========================================================
def plot_activity(per_minute: pd.DataFrame, session_name: str, save_path: Path):
    g = per_minute[
        (per_minute["session"] == session_name)
        & (per_minute["lane"] == "mean_active_lanes")
    ].sort_values("minute_idx").copy()

    if g.empty:
        return

    g["mean_speed_smooth"] = g["mean_speed_px_s"].rolling(ROLLING_MINUTES, center=True, min_periods=1).mean()
    g["distance_smooth"] = g["total_distance_px"].rolling(ROLLING_MINUTES, center=True, min_periods=1).mean()
    g["stop_smooth"] = g["stop_ratio"].rolling(ROLLING_MINUTES, center=True, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(g["minute_idx"], g["mean_speed_px_s"], alpha=0.4, label="raw")
    axes[0].plot(g["minute_idx"], g["mean_speed_smooth"], linewidth=2, label="smoothed")
    axes[0].set_ylabel("Mean speed (px/s)")
    axes[0].set_title(f"{session_name} | Activity state")
    axes[0].legend()

    axes[1].plot(g["minute_idx"], g["total_distance_px"], alpha=0.4, label="raw")
    axes[1].plot(g["minute_idx"], g["distance_smooth"], linewidth=2, label="smoothed")
    axes[1].set_ylabel("Distance per min (px)")
    axes[1].legend()

    axes[2].plot(g["minute_idx"], g["stop_ratio"], alpha=0.4, label="raw")
    axes[2].plot(g["minute_idx"], g["stop_smooth"], linewidth=2, label="smoothed")
    axes[2].set_ylabel("Stop ratio")
    axes[2].set_xlabel("Minute")
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_space(per_minute: pd.DataFrame, session_name: str, save_path: Path):
    g = per_minute[
        (per_minute["session"] == session_name)
        & (per_minute["lane"] == "mean_active_lanes")
    ].sort_values("minute_idx").copy()

    if g.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(g["minute_idx"], g["left_ratio"], label="left")
    axes[0].plot(g["minute_idx"], g["middle_ratio"], label="middle")
    axes[0].plot(g["minute_idx"], g["right_ratio"], label="right")
    axes[0].set_ylabel("Occupancy ratio")
    axes[0].set_title(f"{session_name} | Spatial exploration")
    axes[0].legend()

    axes[1].plot(g["minute_idx"], g["zone_entropy_bits"])
    axes[1].set_ylabel("Zone entropy")

    axes[2].plot(g["minute_idx"], g["spatial_coverage"], label="spatial coverage")
    axes[2].plot(g["minute_idx"], g["zone_transitions"], label="zone transitions")
    axes[2].set_ylabel("Exploration metrics")
    axes[2].set_xlabel("Minute")
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stability(per_minute: pd.DataFrame, session_name: str, save_path: Path):
    g = per_minute[
        (per_minute["session"] == session_name)
        & (per_minute["lane"] == "mean_active_lanes")
    ].sort_values("minute_idx").copy()

    if g.empty:
        return

    g["speed_smooth"] = g["mean_speed_px_s"].rolling(ROLLING_MINUTES, center=True, min_periods=1).mean()
    g["entropy_smooth"] = g["zone_entropy_bits"].rolling(ROLLING_MINUTES, center=True, min_periods=1).mean()
    g["speed_dev"] = np.abs(g["mean_speed_px_s"] - g["speed_smooth"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(g["minute_idx"], g["mean_speed_px_s"], alpha=0.35, label="raw speed")
    axes[0].plot(g["minute_idx"], g["speed_smooth"], linewidth=2, label="rolling mean speed")
    axes[0].set_ylabel("Speed")
    axes[0].set_title(f"{session_name} | Behavioural stability")
    axes[0].legend()

    axes[1].plot(g["minute_idx"], g["speed_dev"])
    axes[1].set_ylabel("|raw - rolling mean|")

    axes[2].plot(g["minute_idx"], g["zone_entropy_bits"], alpha=0.35, label="raw entropy")
    axes[2].plot(g["minute_idx"], g["entropy_smooth"], linewidth=2, label="rolling mean entropy")
    axes[2].set_ylabel("Entropy")
    axes[2].set_xlabel("Minute")
    axes[2].legend()

    for ax in axes:
        ax.axvline(10, linestyle="--", alpha=0.4)
        ax.axvline(20, linestyle="--", alpha=0.4)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main analysis
# =========================================================
def find_tracking_files(root_dir: Path):
    return sorted(root_dir.rglob(TRACKING_FILENAME))


def analyze_one_session(csv_path: Path):
    session_name = csv_path.parent.name
    print(f"\n[INFO] Analyzing session: {session_name}")

    df = load_tracking_csv(csv_path)
    if df.empty:
        print(f"[WARNING] Empty usable data after truncation: {csv_path}")
        return None

    df = normalize_lane_coordinates(df)
    df = compute_framewise_speed(df)

    df, cleaning_summary = remove_false_detections(df)

    df = normalize_lane_coordinates(df)
    df = compute_framewise_speed(df)
    df = add_space_features(df)

    lane_summary = summarize_lane_session(df, session_name)
    per_minute = compute_per_minute_metrics(df, session_name)
    stability = compute_stability_summary(per_minute, session_name)
    session_summary = summarize_session(lane_summary, stability, cleaning_summary)

    return {
        "lane_summary": lane_summary,
        "per_minute": per_minute,
        "session_summary": session_summary,
        "cleaning_summary": cleaning_summary,
    }


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIG_DIR)

    csv_files = find_tracking_files(ROOT_DIR)
    if not csv_files:
        print(f"[ERROR] No {TRACKING_FILENAME} found under {ROOT_DIR}")
        return

    print(f"[INFO] Found {len(csv_files)} sessions")

    all_lane_summary = []
    all_per_minute = []
    all_session_summary = []
    all_cleaning_summary = []

    for csv_path in csv_files:
        try:
            result = analyze_one_session(csv_path)
            if result is None:
                continue

            lane_summary = result["lane_summary"]
            per_minute = result["per_minute"]
            session_summary = result["session_summary"]
            cleaning_summary = result["cleaning_summary"]

            session_name = session_summary["session"].iloc[0]

            all_lane_summary.append(lane_summary)
            all_per_minute.append(per_minute)
            all_session_summary.append(session_summary)
            all_cleaning_summary.append(cleaning_summary)

            plot_activity(per_minute, session_name, FIG_DIR / f"{session_name}_activity.png")
            plot_space(per_minute, session_name, FIG_DIR / f"{session_name}_space.png")
            plot_stability(per_minute, session_name, FIG_DIR / f"{session_name}_stability.png")

        except Exception as e:
            print(f"[ERROR] Failed for {csv_path}: {e}")

    if not all_session_summary:
        print("[ERROR] No session successfully analyzed.")
        return

    lane_summary_df = pd.concat(all_lane_summary, ignore_index=True)
    per_minute_df = pd.concat(all_per_minute, ignore_index=True)
    session_summary_df = pd.concat(all_session_summary, ignore_index=True)
    cleaning_summary_df = pd.concat(all_cleaning_summary, ignore_index=True)

    lane_summary_df.to_csv(OUTPUT_DIR / "lane_summary.csv", index=False)
    per_minute_df.to_csv(OUTPUT_DIR / "per_minute_metrics.csv", index=False)
    session_summary_df.to_csv(OUTPUT_DIR / "session_summary.csv", index=False)
    cleaning_summary_df.to_csv(OUTPUT_DIR / "cleaning_summary.csv", index=False)

    print("\n[INFO] Analysis finished.")
    print(f"[INFO] Output folder: {OUTPUT_DIR}")
    print("[INFO] session_summary.csv")
    print("[INFO] lane_summary.csv")
    print("[INFO] per_minute_metrics.csv")
    print("[INFO] cleaning_summary.csv")
    print("[INFO] figures/")


if __name__ == "__main__":
    main()
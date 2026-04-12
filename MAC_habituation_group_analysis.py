#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAC_habituation_group_analysis.py

Group-level analysis for repeated zebrafish habituation sessions.

Input:
    /Volumes/Raina/habituation/habituation_analysis_outputs/
        session_summary.csv
        per_minute_metrics.csv

Output:
    /Volumes/Raina/habituation/habituation_group_outputs/
        session_order_summary.csv
        minute_level_mean_active_lanes.csv
        figures/
            session_trends.png
            minute_curves_speed.png
            minute_curves_entropy.png
            minute_curves_stop_ratio.png
            habituation_index_trends.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
ROOT_DIR = Path("/Volumes/Raina/habituation")
INPUT_DIR = ROOT_DIR / "habituation_analysis_outputs"
OUTPUT_DIR = ROOT_DIR / "habituation_group_outputs"
FIG_DIR = OUTPUT_DIR / "figures"


# =========================================================
# Utility
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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


def add_session_order(df: pd.DataFrame, session_col: str = "session") -> pd.DataFrame:
    out = df.copy()
    ordered_sessions = sorted(out[session_col].dropna().unique())
    order_map = {s: i + 1 for i, s in enumerate(ordered_sessions)}
    out["session_order"] = out[session_col].map(order_map)
    return out


# =========================================================
# Load
# =========================================================
def load_inputs():
    session_summary_path = INPUT_DIR / "session_summary.csv"
    per_minute_path = INPUT_DIR / "per_minute_metrics.csv"

    if not session_summary_path.exists():
        raise FileNotFoundError(f"Missing file: {session_summary_path}")
    if not per_minute_path.exists():
        raise FileNotFoundError(f"Missing file: {per_minute_path}")

    session_summary = pd.read_csv(session_summary_path)
    per_minute = pd.read_csv(per_minute_path)

    session_summary = add_session_order(session_summary, "session")
    per_minute = add_session_order(per_minute, "session")

    return session_summary, per_minute


# =========================================================
# Derived metrics
# =========================================================
def build_session_order_summary(session_summary: pd.DataFrame) -> pd.DataFrame:
    df = session_summary.copy().sort_values("session_order")

    df["speed_habituation_index"] = (
        (df["early_mean_speed_px_s"] - df["late_mean_speed_px_s"])
        / df["early_mean_speed_px_s"]
    )

    df["speed_drop_absolute"] = df["early_mean_speed_px_s"] - df["late_mean_speed_px_s"]

    df["exploration_drop_entropy"] = np.nan
    if "zone_entropy_slope_per_min" in df.columns:
        df["exploration_drop_entropy"] = -df["zone_entropy_slope_per_min"]

    return df


# =========================================================
# Plotting
# =========================================================
def plot_session_trends(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    axes = axes.flatten()

    metrics = [
        ("mean_speed_px_s", "Mean speed (px/s)"),
        ("total_distance_px", "Total distance (px)"),
        ("stop_ratio", "Stop ratio"),
        ("zone_entropy_bits", "Zone entropy"),
        ("spatial_coverage", "Spatial coverage"),
        ("zone_transitions_per_min", "Zone transitions/min"),
    ]

    for ax, (col, label) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        ax.plot(df["session_order"], df[col], marker="o")
        slope = fit_slope(df["session_order"], df[col])
        ax.set_title(f"{label}\ntrend slope = {slope:.4f}" if pd.notna(slope) else label)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    for ax in axes:
        ax.set_xlabel("Session order")

    fig.suptitle("Repeated habituation: session-level trends", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_minute_curves(per_minute: pd.DataFrame, value_col: str, ylabel: str, save_path: Path):
    g = per_minute[per_minute["lane"] == "mean_active_lanes"].copy()
    g = g.sort_values(["session_order", "minute_idx"])

    fig, ax = plt.subplots(figsize=(11, 7))

    for session_name, sub in g.groupby("session"):
        sub = sub.sort_values("minute_idx")
        session_order = sub["session_order"].iloc[0]
        ax.plot(
            sub["minute_idx"],
            sub[value_col],
            label=f"S{session_order}",
            alpha=0.8
        )

    ax.set_xlabel("Minute")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Repeated habituation: {ylabel} across sessions")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_habituation_indices(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(df["session_order"], df["speed_habituation_index"], marker="o")
    slope1 = fit_slope(df["session_order"], df["speed_habituation_index"])
    axes[0].set_title(
        f"Speed habituation index\ntrend slope = {slope1:.4f}" if pd.notna(slope1) else "Speed habituation index"
    )
    axes[0].set_ylabel("(early - late) / early")
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["session_order"], df["speed_drop_absolute"], marker="o")
    slope2 = fit_slope(df["session_order"], df["speed_drop_absolute"])
    axes[1].set_title(
        f"Absolute speed drop within session\ntrend slope = {slope2:.4f}" if pd.notna(slope2) else "Absolute speed drop within session"
    )
    axes[1].set_ylabel("early - late speed")
    axes[1].set_xlabel("Session order")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIG_DIR)

    session_summary, per_minute = load_inputs()

    session_order_summary = build_session_order_summary(session_summary)
    minute_level_mean = (
        per_minute[per_minute["lane"] == "mean_active_lanes"]
        .sort_values(["session_order", "minute_idx"])
        .copy()
    )

    session_order_summary.to_csv(OUTPUT_DIR / "session_order_summary.csv", index=False)
    minute_level_mean.to_csv(OUTPUT_DIR / "minute_level_mean_active_lanes.csv", index=False)

    plot_session_trends(session_order_summary, FIG_DIR / "session_trends.png")
    plot_minute_curves(
        minute_level_mean,
        value_col="mean_speed_px_s",
        ylabel="Mean speed (px/s)",
        save_path=FIG_DIR / "minute_curves_speed.png",
    )
    plot_minute_curves(
        minute_level_mean,
        value_col="zone_entropy_bits",
        ylabel="Zone entropy",
        save_path=FIG_DIR / "minute_curves_entropy.png",
    )
    plot_minute_curves(
        minute_level_mean,
        value_col="stop_ratio",
        ylabel="Stop ratio",
        save_path=FIG_DIR / "minute_curves_stop_ratio.png",
    )
    plot_habituation_indices(session_order_summary, FIG_DIR / "habituation_index_trends.png")

    print("[INFO] Group analysis finished.")
    print(f"[INFO] Output folder: {OUTPUT_DIR}")
    print("[INFO] session_order_summary.csv")
    print("[INFO] minute_level_mean_active_lanes.csv")
    print("[INFO] figures/")


if __name__ == "__main__":
    main()

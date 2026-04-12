#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAC_habituation_simple_plots.py

Simplified plotting script for habituation results.

Inputs:
    /Volumes/Raina/habituation/habituation_group_outputs/session_order_summary.csv
    /Volumes/Raina/habituation/habituation_group_outputs/minute_level_mean_active_lanes.csv

Outputs:
    /Volumes/Raina/habituation/habituation_simple_plots/
        01_session_overview_bars.png
        02_phase_summary_bars.png
        03_first3_vs_last3_bars.png
        04_barrier_vs_no_barrier_bars.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
ROOT_DIR = Path("/Volumes/Raina/habituation")
INPUT_DIR = ROOT_DIR / "habituation_group_outputs"
OUTPUT_DIR = ROOT_DIR / "habituation_simple_plots"

SESSION_SUMMARY_FILE = INPUT_DIR / "session_order_summary.csv"
MINUTE_LEVEL_FILE = INPUT_DIR / "minute_level_mean_active_lanes.csv"


# =========================================================
# User settings
# =========================================================
BARRIER_SESSIONS = [4, 5, 7]

PHASE_BINS = {
    "Early": list(range(0, 10)),
    "Middle": list(range(10, 20)),
    "Late": list(range(20, 30)),
}


# =========================================================
# Utility
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def mean_sem(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan
    mean = s.mean()
    sem = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
    return mean, sem


def add_barrier_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["barrier_group"] = np.where(
        out["session_order"].isin(BARRIER_SESSIONS),
        "Barrier",
        "No barrier"
    )
    return out


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Load
# =========================================================
def load_data():
    if not SESSION_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SESSION_SUMMARY_FILE}")
    if not MINUTE_LEVEL_FILE.exists():
        raise FileNotFoundError(f"Missing file: {MINUTE_LEVEL_FILE}")

    session_df = pd.read_csv(SESSION_SUMMARY_FILE)
    minute_df = pd.read_csv(MINUTE_LEVEL_FILE)

    session_df = add_barrier_flag(session_df)
    minute_df = add_barrier_flag(minute_df)

    return session_df, minute_df


# =========================================================
# Phase summary
# =========================================================
def build_phase_summary(minute_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for phase_name, minute_list in PHASE_BINS.items():
        sub = minute_df[minute_df["minute_idx"].isin(minute_list)].copy()

        grouped = (
            sub.groupby(["session", "session_order", "barrier_group"], as_index=False)
            .agg(
                mean_speed_px_s=("mean_speed_px_s", "mean"),
                total_distance_px=("total_distance_px", "mean"),
                stop_ratio=("stop_ratio", "mean"),
                zone_entropy_bits=("zone_entropy_bits", "mean"),
                zone_transitions=("zone_transitions", "mean"),
            )
        )
        grouped["phase"] = phase_name
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


# =========================================================
# Plot 1
# =========================================================
def plot_session_overview(session_df: pd.DataFrame, save_path: Path):
    session_df = session_df.sort_values("session_order").copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    metrics = [
        ("mean_speed_px_s", "Mean speed"),
        ("total_distance_px", "Total distance"),
        ("stop_ratio", "Stop ratio"),
        ("zone_transitions_per_min", "Zone transitions/min"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        ax.bar(session_df["session_order"].astype(str), session_df[col])
        ax.set_title(title)
        ax.set_xlabel("Session order")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Session level behavioural overview", fontsize=14)
    save_fig(fig, save_path)


# =========================================================
# Plot 2
# =========================================================
def plot_phase_summary(phase_df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    metrics = [
        ("mean_speed_px_s", "Mean speed"),
        ("stop_ratio", "Stop ratio"),
        ("zone_entropy_bits", "Zone entropy"),
        ("zone_transitions", "Zone transitions"),
    ]

    phase_order = ["Early", "Middle", "Late"]
    x = np.arange(len(phase_order))

    for ax, (col, title) in zip(axes, metrics):
        means = []
        sems = []

        for phase in phase_order:
            m, s = mean_sem(phase_df.loc[phase_df["phase"] == phase, col])
            means.append(m)
            sems.append(s)

        ax.bar(x, means, yerr=sems, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_order)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Early vs Middle vs Late", fontsize=14)
    save_fig(fig, save_path)


# =========================================================
# Plot 3
# =========================================================
def plot_first3_vs_last3(session_df: pd.DataFrame, save_path: Path):
    first3 = session_df[session_df["session_order"].isin([1, 2, 3])].copy()
    last3 = session_df[session_df["session_order"].isin([8, 9, 10])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    metrics = [
        ("mean_speed_px_s", "Mean speed"),
        ("total_distance_px", "Total distance"),
        ("stop_ratio", "Stop ratio"),
        ("zone_transitions_per_min", "Zone transitions/min"),
    ]

    x = np.arange(2)
    labels = ["First 3", "Last 3"]

    for ax, (col, title) in zip(axes, metrics):
        m1, s1 = mean_sem(first3[col])
        m2, s2 = mean_sem(last3[col])

        ax.bar(x, [m1, m2], yerr=[s1, s2], capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("First 3 vs Last 3 sessions", fontsize=14)
    save_fig(fig, save_path)


# =========================================================
# Plot 4
# =========================================================
def plot_barrier_vs_no_barrier(session_df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    metrics = [
        ("mean_speed_px_s", "Mean speed"),
        ("total_distance_px", "Total distance"),
        ("stop_ratio", "Stop ratio"),
        ("zone_transitions_per_min", "Zone transitions/min"),
    ]

    x = np.arange(2)
    labels = ["No barrier", "Barrier"]

    no_barrier = session_df[session_df["barrier_group"] == "No barrier"].copy()
    barrier = session_df[session_df["barrier_group"] == "Barrier"].copy()

    for ax, (col, title) in zip(axes, metrics):
        m1, s1 = mean_sem(no_barrier[col])
        m2, s2 = mean_sem(barrier[col])

        ax.bar(x, [m1, m2], yerr=[s1, s2], capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Barrier vs No barrier", fontsize=14)
    save_fig(fig, save_path)


# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    session_df, minute_df = load_data()
    phase_df = build_phase_summary(minute_df)

    plot_session_overview(
        session_df,
        OUTPUT_DIR / "01_session_overview_bars.png"
    )

    plot_phase_summary(
        phase_df,
        OUTPUT_DIR / "02_phase_summary_bars.png"
    )

    plot_first3_vs_last3(
        session_df,
        OUTPUT_DIR / "03_first3_vs_last3_bars.png"
    )

    plot_barrier_vs_no_barrier(
        session_df,
        OUTPUT_DIR / "04_barrier_vs_no_barrier_bars.png"
    )

    print("[INFO] Simple plots finished.")
    print(f"[INFO] Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


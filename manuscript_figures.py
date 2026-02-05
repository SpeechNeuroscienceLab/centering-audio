#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import *
from subject_analysis import Dataset
import compute_ld_centering
import figure


def only_overshoot(trials: pd.DataFrame) -> pd.DataFrame:
    filtered_trials = trials.copy()
    filtered_trials["Normalized Pitch Movement"] = filtered_trials.apply(
        lambda row: row["Pitch Movement"] * -1
        if row["Tercile"] == "UPPER"
        else row["Pitch Movement"],
        axis=1,
    )

    filtered_trials = filtered_trials[
        np.logical_and(
            filtered_trials["Normalized Pitch Movement"]
            >= np.abs(filtered_trials["Starting Pitch (Cents)"]),
            filtered_trials["Ending Pitch (Cents)"] != 0,
        )
    ]

    filtered_trials.reset_index(inplace=True, drop=True)
    filtered_trials.drop(columns=["Normalized Pitch Movement"], inplace=True)
    print(
        f"Removed {trials.shape[0] - filtered_trials.shape[0]} trials by filtering only overshoot"
    )
    return filtered_trials


def only_anticentering(trials: pd.DataFrame) -> pd.DataFrame:
    filtered_trials = trials.copy()
    filtered_trials = filtered_trials[filtered_trials["Centering (Cents)"] < 0]
    filtered_trials.reset_index(inplace=True, drop=True)
    print(f"Removed {trials.shape[0] - filtered_trials.shape[0]} trials by filtering")
    return filtered_trials


def only_centering(trials: pd.DataFrame) -> pd.DataFrame:
    filtered_trials = trials.copy()
    filtered_trials = filtered_trials[filtered_trials["Centering (Cents)"] > 0]
    filtered_trials.reset_index(inplace=True, drop=True)
    print(
        f"Removed {trials.shape[0] - filtered_trials.shape[0]} trials by filtering out of {trials.shape[0]}"
    )
    return filtered_trials


RESEARCH_DIR = Path("~/Documents/UCSF")
COHORT = "Combined/"
EXPERIMENT = "All trials"
FORCE_ANALYSIS = True
DPI = 300

PLOT_SETTINGS = {
    "plot_order": ["Controls", "LD Patients"],
    "colormap": {
        "Controls": {
            "UPPER": "#FFDAB9",
            "LOWER": "#F4A460",
        },
        "LD Patients": {
            "UPPER": "#87CEFA",
            "LOWER": "#4682B4",
        },
    },
    "error-color": "black",
    "line-width": 1.5,
    "error-cap-size": 10,
    "error-line-style": "",
    "font_size": 10,
    "font_family": "serif",
    "motion_lw": 2.0,
    "motion_alpha": 0.75,
    "annotation_padding": 2,
}

TRIMMING_PIPELINE = [
    trim_by_subject_trial_count,
    trim_by_group_initial_pitch_distribution,
    (
        trim_by_subject_name,
        dict(exclude=[("LD Patients", "20170516"), ("LD Patients", "20170307")]),
    ),
    rename_subjects_by_group,
]


def setup_paths():
    input_path = RESEARCH_DIR / "datasets" / COHORT
    output_path = RESEARCH_DIR / "results" / COHORT / EXPERIMENT
    cache_path = output_path / "cache"
    return input_path, output_path, cache_path


def load_dataset():
    input_path, _, _ = setup_paths()
    ld = Dataset(
        str(input_path / "LDPatients.mat"), "LD_P50_Patients_12_5_24"
    ) + Dataset(str(input_path / "ADLDP50Controls.mat"), "AD_LD_P50_Controls_12_5_24")
    return ld


def setup_analysis():
    _, output_path, cache_path = setup_paths()
    ld = load_dataset()
    demographics = pd.DataFrame({"Group Name": [], "Subject Name": []})

    if FORCE_ANALYSIS or not (cache_path / "centering_data.csv").is_file():
        compute_ld_centering.gen_centering_csv(
            dataset=ld, OUTPUT_PATH=output_path, CACHE_PATH=cache_path
        )

    if FORCE_ANALYSIS or not (
        (cache_path / "trimmed_dataset.csv").is_file()
        and (cache_path / "peripheral_dataset.csv").is_file()
    ):
        compute_ld_centering.gen_centering_analysis(
            demographics=demographics,
            CACHE_PATH=cache_path,
            trimming_pipeline=TRIMMING_PIPELINE,
            filter_pipeline=[],
        )


def create_centering_centering_plot(output_dir):
    _, _, cache_path = setup_paths()
    output_dataset = pd.read_csv(cache_path / "peripheral_dataset.csv")
    output_dataset["Centering Class"] = output_dataset["Centering (Cents)"].apply(
        lambda x: "Centering" if x > 0 else "Anti-centering"
    )
    output_dataset["Overshoot"] = (
        output_dataset["Normalized Pitch Movement"]
    ) > np.abs(output_dataset["Starting Pitch (Cents)"])

    centering_dataset = output_dataset[
        output_dataset["Centering Class"] == "Centering"
    ].copy()

    fig, ax = plt.subplots(1, 1, dpi=DPI)
    ax.set_title("Centering in Centering Trials")
    figure.group_tercile_bars(
        centering_dataset,
        fig,
        PLOT_SETTINGS
        | {
            "label_alias": {"UPPER": "Upper", "LOWER": "Lower"},
            "bar_spacing": 1.0,
            "group_spacing": 0.2,
        },
        legend_loc="upper right",
    )

    figure.significance_bar(ax, 0.5, 2.7, max(ax.get_ylim()) + 2, display_string="***")
    ax.set_ylim(0, 60)

    output_file = output_dir / "centering_centering.png"
    fig.savefig(output_file)
    print(f"Saved {output_file}")
    plt.close()


def create_centering_overshoot_plot(output_dir):
    _, _, cache_path = setup_paths()
    output_dataset = pd.read_csv(cache_path / "peripheral_dataset.csv")
    output_dataset["Centering Class"] = output_dataset["Centering (Cents)"].apply(
        lambda x: "Centering" if x > 0 else "Anti-centering"
    )
    output_dataset["Overshoot"] = (
        output_dataset["Normalized Pitch Movement"]
    ) > np.abs(output_dataset["Starting Pitch (Cents)"])

    centering_dataset = output_dataset[
        output_dataset["Centering Class"] == "Centering"
    ].copy()

    overshoot_dataset = centering_dataset[centering_dataset["Overshoot"]].copy()
    overshoot_dataset["Overshoot Magnitude"] = (
        np.abs(
            (overshoot_dataset["Normalized Pitch Movement"])
            - (overshoot_dataset["Centering (Cents)"])
        )
        / 2
    )

    fig, ax = plt.subplots(1, 1, dpi=DPI)
    ax.set_title("Centering in Centering and Overshoot Trials")
    figure.group_tercile_bars(
        overshoot_dataset,
        fig,
        PLOT_SETTINGS
        | {
            "label_alias": {"UPPER": "Upper", "LOWER": "Lower"},
            "bar_spacing": 1.0,
            "group_spacing": 0.2,
        },
        legend_loc="upper right",
    )

    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1, 1.05))

    figure.significance_bar(ax, 0.5, 2.7, max(ax.get_ylim()) + 3, display_string="*")

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_ylim(0, 60)

    output_file = output_dir / "centering_overshoot.png"
    fig.savefig(output_file)
    print(f"Saved {output_file}")
    plt.close()


def create_anticentering_anticentering_plot(output_dir):
    _, _, cache_path = setup_paths()
    output_dataset = pd.read_csv(cache_path / "peripheral_dataset.csv")
    output_dataset["Centering Class"] = output_dataset["Centering (Cents)"].apply(
        lambda x: "Centering" if x > 0 else "Anti-centering"
    )

    anticentering_dataset = output_dataset[
        output_dataset["Centering Class"] == "Anti-centering"
    ].copy()

    fig, ax = plt.subplots(1, 1, dpi=DPI)
    ax.set_title("AntiCentering in AntiCentering Trials")
    figure.group_tercile_bars(
        anticentering_dataset,
        fig,
        PLOT_SETTINGS
        | {
            "label_alias": {"UPPER": "Upper", "LOWER": "Lower"},
            "bar_spacing": 1.0,
            "group_spacing": 0.2,
        },
        legend_loc="upper right",
    )

    figure.significance_bar(ax, 0, 1, -35, display_string="*")
    figure.significance_bar(ax, 2.2, 3.2, -35, display_string="")
    figure.significance_bar(ax, 0.5, 2.7, -40, display_string="**")

    ax.set_ylim(-60, 0)
    ax.invert_yaxis()

    output_file = output_dir / "anticentering_anticentering.png"
    fig.savefig(output_file)
    print(f"Saved {output_file}")
    plt.close()


def create_anticentering_overshoot_plot(output_dir):
    _, _, cache_path = setup_paths()
    output_dataset = pd.read_csv(cache_path / "peripheral_dataset.csv")
    output_dataset["Centering Class"] = output_dataset["Centering (Cents)"].apply(
        lambda x: "AntiCentering" if x < 0 else "Centering"
    )
    output_dataset["Overshoot"] = (
        output_dataset["Normalized Pitch Movement"]
    ) > np.abs(output_dataset["Starting Pitch (Cents)"])

    centering_dataset = output_dataset[
        output_dataset["Centering Class"] == "AntiCentering"
    ].copy()

    overshoot_dataset = centering_dataset[centering_dataset["Overshoot"]].copy()
    overshoot_dataset["Overshoot Magnitude"] = (
        np.abs(
            overshoot_dataset["Normalized Pitch Movement"]
            - overshoot_dataset["Centering (Cents)"]
        )
        / 2
    )

    fig, ax = plt.subplots(1, 1, dpi=DPI)
    ax.set_title("AntiCentering in AntiCentering and Overshoot Trials")
    figure.group_tercile_bars(
        overshoot_dataset,
        fig,
        PLOT_SETTINGS
        | {
            "label_alias": {"UPPER": "Upper", "LOWER": "Lower"},
            "bar_spacing": 1.0,
            "group_spacing": 0.2,
        },
        legend_loc="upper right",
    )

    figure.significance_bar(ax, 0.5, 2.7, -41, display_string="*")
    figure.significance_bar(ax, 0, 1, -37, display_string="")
    figure.significance_bar(ax, 2.2, 3.2, -37, display_string="")

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_ylim(-60, 0)
    ax.invert_yaxis()

    output_file = output_dir / "anticentering_overshoot.png"
    fig.savefig(output_file)
    print(f"Saved {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript figures")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument(
        "--centering-centering",
        action="store_true",
        help="Generate centering centering plot",
    )
    parser.add_argument(
        "--centering-overshoot",
        action="store_true",
        help="Generate centering overshoot plot",
    )
    parser.add_argument(
        "--anticentering-anticentering",
        action="store_true",
        help="Generate anticentering anticentering plot",
    )
    parser.add_argument(
        "--anticentering-overshoot",
        action="store_true",
        help="Generate anticentering overshoot plot",
    )
    parser.add_argument()

    args = parser.parse_args()

    # Check if any valid arguments were provided
    has_valid_args = (
        args.all
        or args.centering_centering
        or args.centering_overshoot
        or args.anticentering_anticentering
        or args.anticentering_overshoot
        or args.pitch_movement
    )

    if not has_valid_args:
        parser.print_help()
        return

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    setup_analysis()

    if args.all:
        create_centering_centering_plot(output_dir)
        create_centering_overshoot_plot(output_dir)
        create_anticentering_anticentering_plot(output_dir)
        create_anticentering_overshoot_plot(output_dir)
    else:
        if args.centering_centering:
            create_centering_centering_plot(output_dir)
        if args.centering_overshoot:
            create_centering_overshoot_plot(output_dir)
        if args.anticentering_anticentering:
            create_anticentering_anticentering_plot(output_dir)
        if args.anticentering_overshoot:
            create_anticentering_overshoot_plot(output_dir)


if __name__ == "__main__":
    main()

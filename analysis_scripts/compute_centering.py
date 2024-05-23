import time

import matplotlib
import numpy as np

from matplotlib import pyplot as plt

start_time = time.time()

import os
from pathlib import Path

from pipeline import pipeline
from analysis import *
import figure

# Global configuration settings
RESEARCH_DIR = Path(os.path.realpath(__file__)).parent.parent

COHORT = "AD"

INPUT_PATH = RESEARCH_DIR / "datasets" / COHORT
OUTPUT_PATH = RESEARCH_DIR / "results" / COHORT
CACHE_PATH = RESEARCH_DIR / "results" / COHORT / "cache"

# Run configuration settings

FORCE_ANALYSIS = False

# Default/Global plotting settings
DPI = 300

plot_settings = {
    "plot_order": ["Controls", "AD Patients"],
    "colormap": None,  # TODO: set a default behavior

    "error-color": "black",
    "line-width": 1.5,
    "error-cap-size": 10,
    "error-line-style": '',
    "font_size": 13
}

font = {'family': 'sans-serif',
        'size': plot_settings["font_size"]}
matplotlib.rc('font', **font)

# Setup output directories (if they don't exist already)
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

if FORCE_ANALYSIS or not (Path(CACHE_PATH / "trimmed_dataset.csv").is_file()
                          and Path(CACHE_PATH / "peripheral_dataset.csv").is_file()):
    print("Reanalyzing dataset.")

    centering_data = pd.read_csv(INPUT_PATH / "centering_data.csv")
    demographics_dataset = pd.read_csv(INPUT_PATH/"demographics_data.csv")

    # merge demographics and centering datasets
    centering_data["CDR"] = "NAN"
    for group in centering_data["Group Name"].unique():
        subjects = centering_data[centering_data["Group Name"] == group]["Subject Name"].unique()
        for subject in subjects:
            try:
                CDR = float(demographics_dataset[(demographics_dataset["Group Name"] == group) & (demographics_dataset["Subject Name"] == subject)]["CDR"].iloc[0])
            except IndexError:
                print(f"WARNING: Group {group} and Subject {subject} does not have available demographics")
            centering_data.loc[((centering_data["Group Name"] == group) & (centering_data["Subject Name"] == subject)), "CDR"] = CDR

    # useful to know which subject corresponds to each subject index
    subject_alias_table = {group: centering_data[centering_data["Group Name"] == group]["Subject Name"].unique()
                           for group in centering_data["Group Name"].unique()}

    trimmed_dataset = pipeline(centering_data, [
        trim_by_subject_trial_count,
        trim_by_group_initial_pitch_distribution,
        (trim_by_subject_name,
         dict(exclude=[("AD Patients", "20160128_1")])  # AD cohort
         ),
        rename_subjects_by_group  # is this strictly necessary?
    ])

    # add the tercile of each trial
    trimmed_dataset["Tercile"] = compute_trial_tercile(trimmed_dataset)
    trimmed_dataset["Pitch Movement"] = trimmed_dataset["Starting Pitch (Cents)"] - trimmed_dataset[
        "Ending Pitch (Cents)"]

    peripheral_dataset = trimmed_dataset[trimmed_dataset["Tercile"] != "CENTRAL"]

    # save to disk
    trimmed_dataset.to_csv(CACHE_PATH / "trimmed_dataset.csv", index=False)
    peripheral_dataset.to_csv(CACHE_PATH / "peripheral_dataset.csv", index=False)

    print(f"Analysis Completed in {round(time.time() - start_time, 2)} seconds")
else:
    trimmed_dataset = pd.read_csv(CACHE_PATH / "trimmed_dataset.csv")
    peripheral_dataset = pd.read_csv(CACHE_PATH / "peripheral_dataset.csv")

    print(f"Analysis results loaded in {round(time.time() - start_time, 2)} seconds")

# Analysis estimated 10x speedup

fig, ax = plt.subplots(figsize=(6, 4.8), dpi=DPI)

figure.group_tercile_centering_bars(peripheral_dataset,
                                    ax,
                                    plot_settings | {
                                        # override the plotting settings here
                                        "colormap": {
                                            "Controls": {
                                                "LOWER": "lightcoral",
                                                "UPPER": "brown"
                                            },
                                            "AD Patients": {
                                                "LOWER": "mediumturquoise",
                                                "UPPER": "darkcyan"
                                            }
                                        },
                                        "label_alias": {
                                            "UPPER": "Lowering",
                                            "LOWER": "Raising"
                                        },
                                        "bar_spacing": 1.0,
                                        "group_spacing": 0.2,
                                    })

fig.savefig(OUTPUT_PATH / "group_tercile_centering_bars.png")

fig, ax = plt.subplots(figsize=(9, 4.8), dpi=DPI)

figure.group_tercile_centering_bars(peripheral_dataset,
                                    ax,
                                    plot_settings | {
                                        # override the plotting settings here
                                        "plot_order": [0.0, 0.5, 1.0],
                                        "bar_spacing": 1.0,
                                        "group_spacing": 0.5,
                                        "colormap": {
                                            0.0: {
                                                "LOWER": "#008000",
                                                "UPPER": "#15B01A"
                                            },
                                            0.5: {
                                                "LOWER": "#FFA500",
                                                "UPPER": "#F97306"
                                            },
                                            1.0: {
                                                "LOWER": "#FF6347",
                                                "UPPER": "#EF4026"
                                            }
                                        },
                                        "label_alias": {
                                            "UPPER": "Lowering",
                                            "LOWER": "Raising"
                                        },
                                    }, group_column="CDR")

fig.savefig(OUTPUT_PATH / "tercile_centering_bars_by_cdr.png")

print(f"Plots generated in {round(time.time() - start_time, 2)} seconds")
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
    "font_size": 14,

    "annotation_padding": 2
}

font = {'family': 'Times New Roman',
        'size': plot_settings["font_size"]}
matplotlib.rc('font', **font)

# Setup output directories (if they don't exist already)
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

if FORCE_ANALYSIS or not (Path(CACHE_PATH / "trimmed_dataset.csv").is_file()
                          and Path(CACHE_PATH / "peripheral_dataset.csv").is_file()):
    print("Reanalyzing dataset.")

    centering_data = pd.read_csv(INPUT_PATH / "centering_data.csv")
    demographics_dataset = pd.read_csv(INPUT_PATH / "demographics_data.csv")

    # merge demographics and centering datasets
    centering_data["CDR"] = "NAN"
    for group in centering_data["Group Name"].unique():
        subjects = centering_data[centering_data["Group Name"] == group]["Subject Name"].unique()
        for subject in subjects:
            try:
                CDR = float(demographics_dataset[(demographics_dataset["Group Name"] == group) & (
                        demographics_dataset["Subject Name"] == subject)]["CDR"].iloc[0])
            except IndexError:
                print(f"WARNING: Group {group} and Subject {subject} does not have available demographics")
            centering_data.loc[
                ((centering_data["Group Name"] == group) & (centering_data["Subject Name"] == subject)), "CDR"] = CDR

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
    trimmed_dataset["Pitch Movement"] = (trimmed_dataset["Ending Pitch (Cents)"]
                                         - trimmed_dataset["Starting Pitch (Cents)"])

    peripheral_dataset = trimmed_dataset[trimmed_dataset["Tercile"] != "CENTRAL"].reset_index()

    # save to disk
    trimmed_dataset.to_csv(CACHE_PATH / "trimmed_dataset.csv", index=False)
    peripheral_dataset.to_csv(CACHE_PATH / "peripheral_dataset.csv", index=False)

    print(f"Analysis Completed in {round(time.time() - start_time, 2)} seconds")
else:
    trimmed_dataset = pd.read_csv(CACHE_PATH / "trimmed_dataset.csv")
    peripheral_dataset = pd.read_csv(CACHE_PATH / "peripheral_dataset.csv")

    print(f"Analysis results loaded in {round(time.time() - start_time, 2)} seconds")

# Analysis estimated 10x speedup

fig = plt.figure(figsize=(5.0, 4.8), dpi=DPI)
fig.add_axes((0, 0, 1, 1))

figure.group_tercile_bars(peripheral_dataset,
                          fig,
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

# add significance annotation manually
figure.mark_significance_bar(peripheral_dataset, fig, (("AD Patients", "UPPER"), ("AD Patients", "LOWER")),
                             plot_settings | {
                                 "bar_spacing": 1.0,
                                 "group_spacing": 0.2
                             }, display_string="*")

fig.savefig(OUTPUT_PATH / "group_tercile_centering_bars.png", bbox_inches='tight')

fig = plt.figure(figsize=(8, 4.8), dpi=DPI)
fig.add_axes((0, 0, 1, 1))

figure.group_tercile_bars(peripheral_dataset,
                          fig,
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

figure.mark_significance_bar(peripheral_dataset, fig,
                             ((1.0, "UPPER"), (1.0, "LOWER")),
                             plot_settings | {
                                 "bar_spacing": 1.0,
                                 "group_spacing": 0.5,
                                 "plot_order": [0.0, 0.5, 1.0],
                             }, group_column="CDR", display_string="***",
                             height=-3)  # manual height adjustment

figure.mark_significance_bar(peripheral_dataset, fig,
                             ((1.0, "UPPER"), (0.5, "UPPER")),
                             plot_settings | {
                                 "bar_spacing": 1.0,
                                 "group_spacing": 0.5,
                                 "plot_order": [0.0, 0.5, 1.0],
                             }, group_column="CDR", display_string="*",
                             height=0)  # manual height adjustment

fig.savefig(OUTPUT_PATH / "tercile_centering_bars_by_cdr.png", bbox_inches='tight')

print(f"Plots generated in {round(time.time() - start_time, 2)} seconds")

output_dataset = peripheral_dataset.rename(columns={
    "Group Name": "Group",
    "Subject Name": "Subject",
    "Starting Pitch (Cents)": "InitialPitch",
    "Ending Pitch (Cents)": "MidtrialPitch",
    "Centering (Cents)": "Centering",
    "Pitch Movement": "PitchMovement",
})

output_dataset["Trial"] = compute_subject_trial_indices(output_dataset, indexing=("Group", "Subject"))

output_dataset.to_csv(OUTPUT_PATH / "output_dataset.csv", index=False)

print(f"Output dataset generated in {round(time.time() - start_time, 2)} seconds")

# Ad-hoc tasks

# Compute trial count by subject
# for subject in trimmed_dataset["Subject Name"].unique():
#     print(f"Subject {subject} \t {trimmed_dataset[trimmed_dataset['Subject Name'] == subject].shape[0]} trials")
#
# === Results ===
# Patients: 70.77 +- 38.94
# Controls: 122.88 +- 18.91
# Combined: 99.52 +- 39.21

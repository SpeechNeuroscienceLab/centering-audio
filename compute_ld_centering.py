import time

import matplotlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

start_time = time.time()

import os
from pathlib import Path

from pipeline import pipeline
from analysis import *
import figure
from subject_analysis import Dataset, Cohort, Subject

# Global configuration settings
RESEARCH_DIR = Path(os.path.realpath(__file__)).parent.parent

COHORT = "LD"

INPUT_PATH = RESEARCH_DIR / "datasets" / COHORT
OUTPUT_PATH = RESEARCH_DIR / "results" / COHORT
CACHE_PATH = RESEARCH_DIR / "results" / COHORT / "cache"

# Run configuration settings

FORCE_ANALYSIS = True

# Default/Global plotting settings
DPI = 300

plot_settings = {
    "plot_order": ["Controls", "LD Patients"],
    "colormap": {
        "LOWER": "#1f77b4",
        "CENTRAL": "#ff7f0e",
        "UPPER": "#2ca02c"
    },

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

ld = Dataset(str(INPUT_PATH / "raw_dataset_ext.mat"), "ld_dataset_voice_aligned")

if FORCE_ANALYSIS or not (Path(CACHE_PATH) / "centering_data.csv").is_file():
    print("Generating centering CSV.")

    subject_datasets = []

    # generate centering dataset (without trimming)
    for cohort_name, cohort in ld.cohorts.items():
        for subject in cohort.subjects:
            initial_index = (subject.taxis >= 0.0) & (subject.taxis < 0.05)
            midtrial_index = (subject.taxis >= 0.150) & (subject.taxis < 0.200)

            subject_data = pd.DataFrame()

            subject_data["Starting Pitch (Hz)"] = np.mean(subject.trials[:, initial_index], axis=1)
            subject_data["Ending Pitch (Hz)"] = np.mean(subject.trials[:, midtrial_index], axis=1)

            subject_data["Starting Pitch (Cents)"] = 1200 * np.log2(subject_data["Starting Pitch (Hz)"] / np.median(subject_data["Starting Pitch (Hz)"]))
            subject_data["Ending Pitch (Cents)"] = 1200 * np.log2(subject_data["Ending Pitch (Hz)"] / np.median(subject_data["Ending Pitch (Hz)"]))

            subject_data["Centering (Cents)"] = np.abs(subject_data["Starting Pitch (Cents)"]) - np.abs(subject_data["Ending Pitch (Cents)"])
            subject_data["Group Name"] = cohort_name
            subject_data["Subject Name"] = subject.name

            subject_datasets.append(subject_data)

    centering_data = pd.concat(subject_datasets, ignore_index=True)

    # in-place renaming scheme
    centering_data['Group Name'] = centering_data['Group Name'].replace('Patients', 'LD Patients')
    centering_data[["Group Name", "Subject Name", "Starting Pitch (Cents)", "Ending Pitch (Cents)", "Centering (Cents)"]].to_csv(CACHE_PATH / "centering_data.csv", index=False)

    print(f"Analysis Completed in {round(time.time() - start_time, 2)} seconds")

if FORCE_ANALYSIS or not (Path(CACHE_PATH / "trimmed_dataset.csv").is_file()
                          and Path(CACHE_PATH / "peripheral_dataset.csv").is_file()):
    print("Reanalyzing dataset.")

    centering_data = pd.read_csv(CACHE_PATH / "centering_data.csv")

    # useful to know which subject corresponds to each subject index
    subject_alias_table = {group: centering_data[centering_data["Group Name"] == group]["Subject Name"].unique()
                           for group in centering_data["Group Name"].unique()}

    trimmed_dataset = pipeline(centering_data, [
        trim_by_subject_trial_count,
        trim_by_group_initial_pitch_distribution,
        (trim_by_subject_name,
         dict(exclude=[("LD Patients", "20170516"), ("LD Patients", "20170307")])
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
extended_peripheral_dataset = peripheral_dataset.copy()
extended_peripheral_dataset["Pitch Movement Magnitude"] = extended_peripheral_dataset["Pitch Movement"].abs()

# Local-only analysis

fig = plt.figure(figsize=(6, 4), dpi=DPI)
fig.add_axes((0, 0, 1, 1))

figure.group_tercile_centering_bars(peripheral_dataset,
                                    fig,
                                    plot_settings | {
                                        # override the plotting settings here
                                        "colormap": {
                                            "Controls": {
                                                "LOWER": "lightcoral",
                                                "UPPER": "brown"
                                            },
                                            "LD Patients": {
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

figure.mark_significance_bar(peripheral_dataset, fig,
                             (("Controls", "LOWER"), ("Controls", "UPPER")),
                             plot_settings | {
                                 "bar_spacing": 1.0,
                                 "group_spacing": 0.5,
                             }, display_string="*",
                             height=0)  # manual height adjustment

# manual visual adjustments
fig.get_axes()[0].set_ylim((0, 30))

fig.savefig(OUTPUT_PATH / "group_tercile_centering_bars.png", bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(6, 4), dpi=DPI)

figure.group_pitch_distributions(trimmed_dataset, fig, plot_settings | {
    "colormap": {
        "Ending Pitch (Cents)": "darkgreen",
        "Starting Pitch (Cents)": "violet"
    }
})

fig.savefig(OUTPUT_PATH / "group_pitch_dist.png", bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(6, 4), dpi=DPI)

figure.group_pitch_distributions(trimmed_dataset, fig, plot_settings | {
    "colormap": {
        "Ending Pitch (Cents)": "darkgreen",
        "Starting Pitch (Cents)": "violet"
    },
}, smooth_on=True)

fig.savefig(OUTPUT_PATH / "group_pitch_dist_smoothed.png", bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(5.0, 4.8), dpi=DPI)
fig.add_axes((0, 0, 1, 1))

figure.group_tercile_centering_bars(extended_peripheral_dataset,
                                    fig,
                                    plot_settings | {
                                        # override the plotting settings here
                                        "colormap": {
                                            "Controls": {
                                                "LOWER": "lightcoral",
                                                "UPPER": "brown"
                                            },
                                            "LD Patients": {
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
                                    },
                                    quantity_column="Pitch Movement Magnitude")

figure.mark_significance_bar(peripheral_dataset, fig,
                             (("LD Patients", "LOWER"), ("LD Patients", "UPPER")),
                             plot_settings | {
                                 "bar_spacing": 1.0,
                                 "group_spacing": 0.2,
                             }, display_string="*",
                             height=0)  # manual height adjustment

fig.savefig(OUTPUT_PATH / "group_tercile_pitch_movement_bars.png", bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(6., 4.8), dpi=DPI)
fig.add_axes((0, 0, 1, 0.5))
fig.add_axes((0, 0.5, 1, 0.5))
figure.group_centering_distribution(peripheral_dataset,
                                    fig, plot_settings | {},
                                    groups=("LD Patients", "Controls"))
for axis in fig.get_axes():
    axis.set_xlim((-100, 400))
fig.savefig(OUTPUT_PATH / "group_pitch_centering_distribution.png", bbox_inches='tight')
plt.close()

for tercile in ["UPPER", "LOWER"]:
    fig = plt.figure(figsize=(6., 4.8), dpi=DPI)
    fig.add_axes((0, 0, 1, 0.5))
    fig.add_axes((0, 0.5, 1, 0.5))
    figure.group_centering_distribution(peripheral_dataset[peripheral_dataset["Tercile"] == tercile],
                                        fig, plot_settings | {}, title=f"[{tercile.lower()}]",
                                        groups=("LD Patients", "Controls"))
    for axis in fig.get_axes():
        axis.set_xlim((-100, 400))
    fig.savefig(OUTPUT_PATH / f"group_{tercile}_pitch_centering_distribution.png", bbox_inches='tight')
    plt.close()

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

print("== Subject-level analysis ==")

save_directory = str(OUTPUT_PATH / "raw")
Path(save_directory).mkdir(parents=True, exist_ok=True)

for cohort_name in []: #["Patients", "Controls"]:
    for subject_index, _ in enumerate(ld.cohorts[cohort_name].subjects):
        print(f"Plotting {cohort_name} #{subject_index + 1}")
        subject = ld.cohorts[cohort_name].subjects[subject_index]

        trial_count = subject.trials.shape[0]

        # convert trials to cents
        window_medians = np.median(subject.trials, axis=0)
        subject.trials_cents = 1200 * np.log2(subject.trials / window_medians)

        # compute the initial window means for all subjects
        initial_mask = np.logical_and((subject.taxis < 0.05), (subject.taxis >= 0))
        initial_window_means = np.mean(subject.trials_cents[:, initial_mask], axis=1)

        # tercile split of the initial data
        initial_window_order = initial_window_means.argsort()

        # Lower, Central, Upper
        tercile_masks = []
        tercile_labels = ["LOWER", "CENTRAL", "UPPER"]
        for i in range(3):
            trial_indices = initial_window_order[int(i * trial_count / 3): int((i + 1) * trial_count / 3)]

            mask = np.zeros_like(initial_window_order)
            for trial_idx in trial_indices:
                mask[trial_idx] = 1
            tercile_masks.append(mask == 1)

        # plot the average upper and lower and middle
        plt.figure()

        plt.title(f"Mean Productions of {subject.name} ({cohort_name} #{subject_index + 1})")
        plt.xlabel("t (seconds)")
        plt.ylabel("pitch (cents)")

        full_mask = np.logical_and((subject.taxis >= 0.000), (subject.taxis <= 0.200))

        for tercile in range(3):
            tercile_trials = subject.trials_cents[tercile_masks[tercile], :]
            mean_trial = np.mean(tercile_trials, axis=0)
            error_trial = np.std(tercile_trials, axis=0) / np.sqrt(max(mean_trial.shape))
            plt.plot(subject.taxis[full_mask], mean_trial[full_mask], label=f"{tercile_labels[tercile]}")

            plt.fill_between(subject.taxis[full_mask],
                             mean_trial[full_mask] - error_trial[full_mask],
                             mean_trial[full_mask] + error_trial[full_mask],
                             alpha=0.5)

        median_trial = np.median(subject.trials_cents, axis=0)
        plt.plot(subject.taxis[full_mask], median_trial[full_mask],
                 color="black", linestyle="dotted")

        plt.legend()

        plt.savefig(f"{save_directory}/{cohort_name}_{subject_index + 1}.png")
        plt.close()

        # plot all the trials for the subject
        plt.figure()

        plt.title(f"{subject.name} ({cohort_name} #{subject_index + 1}) (All Trials)")
        plt.xlabel("t (seconds)")
        plt.ylabel("pitch (cents)")

        for i, trial in enumerate(subject.trials_cents):
            tercile = 0 if tercile_masks[0][i] == True \
                else 1 if tercile_masks[1][i] == True \
                else 2
            if tercile in [0, 2]:
                plt.plot(subject.taxis[full_mask],
                         trial[full_mask],
                         color=list(plot_settings["colormap"].items())[tercile][1],
                         alpha=0.5)

        median_trial = np.median(subject.trials_cents, axis=0)
        plt.plot(subject.taxis[full_mask], median_trial[full_mask],
                 color="black", linestyle="dotted")

        plt.savefig(f"{save_directory}/all_trials_{cohort_name}_{subject_index + 1}.png")
        plt.close()

print(f"Subject-level datasets generated in {round(time.time() - start_time, 2)} seconds")

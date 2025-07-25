import time

import matplotlib

from matplotlib import pyplot as plt

import os
from pathlib import Path

from pipeline import pipeline
from analysis import *
import figure
from subject_analysis import Dataset, Cohort, Subject


def gen_centering_csv(
        dataset,
        OUTPUT_PATH,
        CACHE_PATH):
    start_time = time.time()
    # alias dataset -> ld
    ld = dataset

    # Setup output directories (if they don't exist already)
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

    print("Generating centering CSV.")

    subject_datasets = []

    # generate centering dataset (without trimming)
    for cohort_name, cohort in ld.cohorts.items():
        for subject in cohort.subjects:
            initial_index = (subject.taxis >= 0.00) & (subject.taxis < 0.05)
            midtrial_index = (subject.taxis >= 0.150) & (subject.taxis < 0.200)

            subject_data = pd.DataFrame()

            median_pitch_hz = np.median(subject.trials, axis=0)

            assert median_pitch_hz.shape[0] == subject.taxis.shape[0]

            subject.trials_cents = 1200 * np.log2(
                np.divide(subject.trials, median_pitch_hz, out=np.zeros_like(subject.trials),
                          where=median_pitch_hz != 0)
            )

            subject_data["Starting Pitch (Cents)"] = np.mean(subject.trials_cents[:, initial_index], axis=1)
            subject_data["Ending Pitch (Cents)"] = np.mean(subject.trials_cents[:, midtrial_index], axis=1)

            subject_data["Centering (Cents)"] = np.abs(subject_data["Starting Pitch (Cents)"]) - np.abs(subject_data["Ending Pitch (Cents)"])

            subject_data["Group Name"] = cohort_name
            subject_data["Subject Name"] = subject.name

            subject_datasets.append(subject_data)

    centering_data = pd.concat(subject_datasets, ignore_index=True)

    # in-place renaming scheme
    centering_data['Group Name'] = centering_data['Group Name'].replace('Patients', 'LD Patients')
    centering_data[
        ["Group Name", "Subject Name", "Starting Pitch (Cents)", "Ending Pitch (Cents)", "Centering (Cents)"]].to_csv(
        CACHE_PATH / "centering_data.csv", index=False)

    print(f"Analysis Completed in {round(time.time() - start_time, 2)} seconds")


def gen_centering_analysis(
        demographics,
        CACHE_PATH,
        trimming_pipeline=None,
        filter_pipeline=None):
    # immutable default argument
    if trimming_pipeline is None:
        trimming_pipeline = [
            trim_by_subject_trial_count,  # remove subjects with <25 trials
            trim_by_group_initial_pitch_distribution,
            # remove trials which have initial pitch deviation >2std from mean
            (trim_by_subject_name,
             dict(exclude=[("LD Patients", "20170516"), ("LD Patients", "20170307")])
             ),
            rename_subjects_by_group  # is this strictly necessary?
        ]

    start_time = time.time()
    print("Reanalyzing dataset.")

    centering_data = pd.merge(pd.read_csv(CACHE_PATH / "centering_data.csv"), demographics,
                              on=["Group Name", "Subject Name"], how="left")

    for column in demographics.columns.values:
        if column not in ["Group Name", "Subject Name"]:
            missing_data = centering_data[pd.isnull(centering_data["Age"])]
            missing_ids = set(zip(missing_data["Group Name"], missing_data["Subject Name"]))
            if len(missing_ids) > 0:
                print(f"Missing {column} data for {len(missing_ids)} subjects")
                for group, subject in missing_ids:
                    print(f"Missing {column} data for {group}/{subject}")

    # useful to know which subject corresponds to each subject index
    subject_alias_table = {group: centering_data[centering_data["Group Name"] == group]["Subject Name"].unique()
                           for group in centering_data["Group Name"].unique()}

    print(f"Unique subjects: {centering_data['Subject Name'].unique().shape[0]}")

    trimmed_dataset = pipeline(centering_data, trimming_pipeline)

    # add the tercile of each trial
    trimmed_dataset["Tercile"] = compute_trial_tercile(trimmed_dataset)
    trimmed_dataset["Pitch Movement"] = (trimmed_dataset["Ending Pitch (Cents)"]
                                         - trimmed_dataset["Starting Pitch (Cents)"])

    if filter_pipeline is not None:
        print("Note: filtering based on filter pipeline")
        trimmed_dataset = pipeline(trimmed_dataset, filter_pipeline)

    peripheral_dataset = trimmed_dataset[trimmed_dataset["Tercile"] != "CENTRAL"].reset_index()

    peripheral_dataset["Normalized Pitch Movement"] = peripheral_dataset.apply(
        lambda row: row["Pitch Movement"] * -1 if row["Tercile"] == "UPPER" else row["Pitch Movement"], axis=1)

    # save to disk
    trimmed_dataset.to_csv(CACHE_PATH / "trimmed_dataset.csv", index=False)
    peripheral_dataset.to_csv(CACHE_PATH / "peripheral_dataset.csv", index=False)

    print(f"Analysis Completed in {round(time.time() - start_time, 2)} seconds")


def gen_group_figures(
        dataset,
        CACHE_PATH,
        OUTPUT_PATH,
        plot_settings,
        DPI
):
    start_time = time.time()
    # Load in the dataset
    trimmed_dataset = pd.read_csv(CACHE_PATH / "trimmed_dataset.csv")
    peripheral_dataset = pd.read_csv(CACHE_PATH / "peripheral_dataset.csv")

    print(f"Analysis results loaded in {round(time.time() - start_time, 2)} seconds")

    # Analysis estimated 10x speedup
    extended_peripheral_dataset = peripheral_dataset.copy()
    extended_peripheral_dataset["Pitch Movement Magnitude"] = extended_peripheral_dataset["Pitch Movement"].abs()

    # Local-only analysis
    font = {'family': plot_settings["font_family"],
            'size': plot_settings["font_size"]}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(6, 4), dpi=DPI)
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

    # figure.mark_significance_bar(peripheral_dataset, fig,
    #                              (("Controls", "LOWER"), ("Controls", "UPPER")),
    #                              plot_settings | {
    #                                  "bar_spacing": 1.0,
    #                                  "group_spacing": 0.5,
    #                              }, display_string="*",
    #                              height=0)  # manual height adjustment

    # manual visual adjustments
    fig.get_axes()[0].set_ylim((0, 50))

    fig.savefig(OUTPUT_PATH / "group_tercile_centering_bars.png", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 4), dpi=DPI)
    figure.group_pitch_distributions(trimmed_dataset, fig, plot_settings | {
        "colormap": {
            "Ending Pitch (Cents)": "salmon",
            "Starting Pitch (Cents)": "brown"
        }
    })

    fig.savefig(OUTPUT_PATH / "group_pitch_dist.png", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(12, 8), dpi=DPI)
    fig.add_axes((0, 0, 1, 0.5))
    fig.add_axes((0, 0.5, 1, 0.5))
    figure.group_subject_centering_overshoot_bars(peripheral_dataset, fig, plot_settings)

    fig.savefig(OUTPUT_PATH / "group_subj_pitch_dist.png", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 6), dpi=DPI)
    fig.add_axes((0, 0, 1, 0.45))
    fig.add_axes((0, 0.55, 1, 0.45))
    figure.group_subject_overshoot_stacked_bars(peripheral_dataset, fig, plot_settings)
    fig.savefig(OUTPUT_PATH / "group_subj_centering_freq.png", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 4), dpi=DPI)

    figure.group_pitch_distributions(trimmed_dataset, fig, plot_settings | {
        "colormap": {
            "Ending Pitch (Cents)": "salmon",
            "Starting Pitch (Cents)": "brown"
        },
    }, smooth_on=True)

    fig.savefig(OUTPUT_PATH / "group_pitch_dist_smoothed.png", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(5.0, 4.8), dpi=DPI)
    fig.add_axes((0, 0, 1, 1))

    figure.group_tercile_bars(extended_peripheral_dataset,
                              fig,
                              plot_settings | {
                                  # override the plotting settings here
                                  "colormap": {
                                      "Controls": {
                                          "LOWER": "navajowhite",
                                          "UPPER": "darkorange"
                                      },
                                      "LD Patients": {
                                          "LOWER": "cornflowerblue",
                                          "UPPER": "lightblue"
                                      }
                                  },
                                  "label_alias": {
                                      "UPPER": "Lowering",
                                      "LOWER": "Raising"
                                  },
                                  "bar_spacing": 1.0,
                                  "group_spacing": 0.2,
                              },
                              quantity_column="Normalized Pitch Movement")

    # figure.mark_significance_bar(peripheral_dataset, fig,
    #                              (("LD Patients", "LOWER"), ("LD Patients", "UPPER")),
    #                              plot_settings | {
    #                                  "bar_spacing": 1.0,
    #                                  "group_spacing": 0.2,
    #                              }, display_string="*",
    #                              height=0)  # manual height adjustment

    fig.savefig(OUTPUT_PATH / "group_tercile_norm_pitch_movement_bars.png", bbox_inches='tight')
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


def gen_subject_figures(
        dataset,
        CACHE_PATH,
        OUTPUT_PATH,
        plot_settings,
        DPI,
        cohorts=["Controls", "Patients"]
):
    start_time = time.time()
    ld = dataset
    # Load in the dataset
    peripheral_dataset = pd.read_csv(CACHE_PATH / "peripheral_dataset.csv")

    save_directory = str(OUTPUT_PATH / "raw")
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    for subject_id in peripheral_dataset["Subject Name"].unique():
        subject_data = peripheral_dataset[peripheral_dataset["Subject Name"] == subject_id]

        print(f"Drawing centering quiver for {subject_id}")

        for tercile in ["UPPER", "LOWER"]:
            fig = plt.figure(figsize=(6., 4.8), dpi=DPI)
            fig.add_axes((0, 0, 1, 1))
            figure.group_centering_distribution(subject_data[subject_data["Tercile"] == tercile],
                                                fig, plot_settings | {},
                                                title=f"[{tercile.lower()} tercile] for {subject_id}",
                                                groups=subject_data["Group Name"].unique())
            for axis in fig.get_axes():
                axis.set_xlim((-100, 400))
            fig.savefig(OUTPUT_PATH / f"raw/{tercile}_{subject_id}_centering_quiver.png", bbox_inches='tight')
            plt.close()

    for cohort_name in cohorts:
        for subject_index, _ in enumerate(ld.cohorts[cohort_name].subjects):
            print(f"Plotting {cohort_name} #{subject_index + 1}")

            # plot the subject distribution plot
            subject = ld.cohorts[cohort_name].subjects[subject_index]

            trial_count = subject.trials.shape[0]

            # convert trials to cents
            window_medians = np.median(subject.trials, axis=0)
            subject.trials_cents = 1200 * np.log2(subject.trials / window_medians)

            # compute the initial window means for all subjects
            initial_mask = np.logical_and((subject.taxis < 0.05), (subject.taxis >= 0.00))
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

            full_mask = np.logical_and((subject.taxis >= 0.00), (subject.taxis <= 0.200))

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
                tercile = 0 if tercile_masks[0][i] is True \
                    else 1 if tercile_masks[1][i] is True \
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

            fig = plt.figure(figsize=(6, 4), dpi=DPI)
            fig.add_axes((0, 0, 1, 1))

            figure.group_tercile_bars(
                peripheral_dataset[peripheral_dataset["Subject Name"] == f"{cohort_name}{subject_index + 1}"],
                fig,
                plot_settings | {
                    # override the plotting settings here
                    "colormap": {
                        "Controls": {
                            "LOWER": "navajowhite",
                            "UPPER": "darkorange"
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

            # figure.mark_significance_bar(peripheral_dataset, fig,
            #                              (("Controls", "LOWER"), ("Controls", "UPPER")),
            #                              plot_settings | {
            #                                  "bar_spacing": 1.0,
            #                                  "group_spacing": 0.5,
            #                              }, display_string="*",
            #                              height=0)  # manual height adjutment

            # manual visual adjustments
            fig.get_axes()[0].set_ylim((0, 30))

            fig.savefig(OUTPUT_PATH / f"raw/group_tercile_centering_bars_{cohort_name}{subject_index + 1}.png",
                        bbox_inches='tight')
            plt.close()

    print(f"Subject-level datasets generated in {round(time.time() - start_time, 2)} seconds")

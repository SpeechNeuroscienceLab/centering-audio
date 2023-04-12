# import statements
import numpy as np
from experiment import Experiment
from pathlib import Path
import figures
import extra


class CenteringAnalysis:
    # contains methods for trimming subjects
    class Trim:
        @staticmethod
        def by_subject_name(experiment: Experiment, exclude: list, inplace=True,
                            indexing=("Group Name", "Subject Name")):
            trimmed_experiment = experiment if inplace else experiment.copy()

            for group, subject in exclude:
                print(f"Manually removing {group}/{subject} from the analysis")
                experiment.df.drop(experiment.df[(experiment.df[indexing[0]] == group) &
                                                 (experiment.df[indexing[1]] == subject)].index, inplace=True)
            trimmed_experiment.update_subjects()

            trimmed_experiment.df.reset_index(inplace=True, drop=True)
            return trimmed_experiment

        @staticmethod
        def by_subject_trial_count(experiment: Experiment, min_trials: int = 25, inplace=True,
                                   indexing=("Group Name", "Subject Name")):
            trimmed_experiment = experiment if inplace else experiment.copy()

            for group, subjects in trimmed_experiment.subjects.items():
                for subject in subjects:
                    subject_trials_df = trimmed_experiment.df[(trimmed_experiment.df[indexing[0]] == group)
                                                              & (trimmed_experiment.df[indexing[1]] == subject)]
                    if subject_trials_df.shape[0] < min_trials:
                        print(f"Excluding {group}/{subject}: {subject_trials_df.shape[0]}/{min_trials}")
                        trimmed_experiment.df.drop(subject_trials_df.index, inplace=True)

            trimmed_experiment.update_subjects()

            trimmed_experiment.df.reset_index(inplace=True, drop=True)
            return trimmed_experiment

        @staticmethod
        def by_subject_initial_pitch_distribution(experiment: Experiment, std_threshold=2, inplace=True,
                                                  indexing=("Group Name", "Subject Name", "Starting Pitch (Cents)")):
            trimmed_experiment = experiment if inplace else experiment.copy()
            for group, subjects in experiment.subjects.items():
                for subject in subjects:
                    subject_trials = trimmed_experiment.df[(trimmed_experiment.df[indexing[0]] == group)
                                                           & (trimmed_experiment.df[indexing[1]] == subject)]

                    subj_mean = np.mean(subject_trials[indexing[2]])
                    subj_std = np.std(subject_trials[indexing[2]])

                    outliers = subject_trials[np.abs(subject_trials[indexing[2]] - subj_mean)
                                              > std_threshold * subj_std]

                    trimmed_experiment.df.drop(outliers.index, inplace=True)

                    print(f"{group}/{subject}: {outliers.shape[0]}/{subject_trials.shape[0]} outliers.")

            trimmed_experiment.df.reset_index(inplace=True, drop=True)
            return trimmed_experiment

        @staticmethod
        def by_group_initial_pitch_distribution(experiment: Experiment, std_threshold=2, inplace=True,
                                                indexing=("Group Name", "Starting Pitch (Cents)")):
            trimmed_experiment = experiment if inplace else experiment.copy()

            for group in experiment.subjects:
                group_trials = trimmed_experiment.df[(trimmed_experiment.df[indexing[0]] == group)]

                subj_mean = np.mean(group_trials[indexing[1]])
                subj_std = np.std(group_trials[indexing[1]])

                outliers = group_trials[np.abs(group_trials[indexing[1]] - subj_mean)
                                        > std_threshold * subj_std]

                trimmed_experiment.df.drop(outliers.index, inplace=True)

                print(f"{group}: {outliers.shape[0]}/{group_trials.shape[0]} outliers.")

            trimmed_experiment.df.reset_index(inplace=True, drop=True)
            return trimmed_experiment

    class ComputeColumn:
        @staticmethod
        def trial_index(experiment: Experiment, inplace=True, indexing=("Group Name", "Subject Name", "Trial Index")):
            updated_experiment = experiment if inplace else experiment.copy()

            # add the trial index column if it doesn't exist already
            if indexing[2] not in updated_experiment.df:
                updated_experiment.df[indexing[2]] = -1

            for group, subjects in updated_experiment.subjects.items():
                for subject in subjects:
                    subject_trials = updated_experiment.df[(updated_experiment.df[indexing[0]] == group)
                                                           & (updated_experiment.df[indexing[1]] == subject)]

                    # set the original index of each trial
                    subject_trials = subject_trials.assign(**{indexing[2]: np.arange(1, subject_trials.shape[0] + 1)})

                    # save changes to the original dataframe
                    updated_experiment.df.update(subject_trials)

            return updated_experiment

        @staticmethod
        def tercile(experiment: Experiment, inplace=True, indexing=("Group Name", "Subject Name",
                                                                    "Starting Pitch (Cents)", "Trial Tercile")):
            updated_experiment = experiment if inplace else experiment.copy()

            # by default, label as blank
            updated_experiment.df[indexing[3]] = ""
            for group, subjects in updated_experiment.subjects.items():
                for subject in subjects:
                    subject_trials = updated_experiment.df[(updated_experiment.df[indexing[0]] == group)
                                                           & (updated_experiment.df[indexing[1]] == subject)].copy()

                    # tercile markers are placed at the 33rd and 66th percentile
                    tercile_markers = np.percentile(subject_trials[indexing[2]],
                                                    100 * np.arange(1, 3) / 3)

                    subject_trials[indexing[3]] = subject_trials.apply(
                        lambda row:
                        "UPPER" if row[indexing[2]] > tercile_markers[1] else
                        "LOWER" if row[indexing[2]] < tercile_markers[0]
                        else "CENTRAL", axis=1)

                    # save changes to the original dataframe
                    updated_experiment.df.update(subject_trials)

            return updated_experiment

    @staticmethod
    def drop_central_trials(experiment: Experiment, inplace=True, indexing="Trial Tercile"):
        updated_experiment = experiment if inplace else experiment.copy()
        # delete all central trials
        updated_experiment.df.drop(updated_experiment.df[
                                       (updated_experiment.df[indexing] == "CENTRAL")].index, inplace=True)
        return updated_experiment

    @staticmethod
    def rename_subjects(experiment: Experiment, inplace=True, indexing="Subject Name"):
        updated_experiment = experiment if inplace else experiment.copy()

        for group, subjects in updated_experiment.subjects.items():
            for idx, subject in enumerate(subjects):
                # for each group, rename the subject
                updated_experiment.df[indexing].replace(subject, (group + str(idx)).replace(" ", ""), inplace=True)

        return updated_experiment


# input args
INPUT_PATH = "/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/analyze_audio.csv"
OUTPUT_PATH = "/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/publication/"

COLUMN_MAP = {
    "Group Name": "Group",
    "Subject Name": "Subject",
    "Starting Pitch (Cents)": "InitialPitch",
    "Ending Pitch (Cents)": "EndingPitch",
    "Centering (Cents)": "Centering",
    "Trial Tercile": "Tercile",
    "Trial Index": "Trial"
}


def main():
    experiment = Experiment.from_csv(INPUT_PATH)
    print(f"Trials Parsed: {experiment.df.shape[0]}")

    run = "manual_exclude"
    std_threshold = 2
    trim_bundle = "group"

    print(f"Using trimming method '{run}' within each {trim_bundle} with std factor {std_threshold}")

    # for auto-trimming
    trimmed_experiment = experiment.copy()
    CenteringAnalysis.Trim.by_subject_trial_count(trimmed_experiment)
    CenteringAnalysis.Trim.by_group_initial_pitch_distribution(trimmed_experiment, std_threshold=std_threshold)

    # manually exclude subjects
    CenteringAnalysis.Trim.by_subject_name(trimmed_experiment, exclude=[("AD Patients", "20160128_1")])

    print(f"Trials trimmed from analysis: {experiment.df.shape[0] - trimmed_experiment.df.shape[0]}")

    # compute terciles
    CenteringAnalysis.ComputeColumn.tercile(trimmed_experiment)

    # Run extra feature: PIDN table generation
    # (must be done before we rename our subjects)
    pidn_df = extra.load_pidn_table(trimmed_experiment)

    # rename subjects
    full_subject_ids = trimmed_experiment.subjects.copy()
    CenteringAnalysis.rename_subjects(trimmed_experiment)
    trimmed_experiment.update_subjects()

    # show rename table for subjects
    print("*Subject Rename Table*")
    for group in trimmed_experiment.subjects:
        print(f"{group}:")
        for subject_index, subject in enumerate(trimmed_experiment.subjects[group]):
            print(f"\t{full_subject_ids[group][subject_index]} -> {subject}")

    # rename columns
    experiment.df.rename(columns=COLUMN_MAP, inplace=True)
    trimmed_experiment.df.rename(columns=COLUMN_MAP, inplace=True)

    # separate into only peripheral trials
    peripheral_experiment = CenteringAnalysis.drop_central_trials(trimmed_experiment,
                                                                  inplace=False, indexing="Tercile")
    # compute trial indices for peripheral trials
    CenteringAnalysis.ComputeColumn.trial_index(peripheral_experiment, inplace=True,
                                                indexing=("Group", "Subject", "Trial"))

    output_folder = OUTPUT_PATH
    # Create the output folder if it doesn't exist already
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    peripheral_experiment.to_csv(f"{output_folder}centering-analysis.csv")
    print(f"Wrote data table to '{output_folder}centering-analysis.csv'")

    for group, subjects in peripheral_experiment.subjects.items():
        print(f"Including {len(subjects)} subjects in group {group} for this analysis.")

    print(f"Generating Figures...")
    figure_list = [
        figures.SampleTrial(
            motion_points=[
                [0, 50, 100, 150, 200],
                [100, 110, 80, 30, 35]
            ]),
        figures.GroupPitchNormal(experiment,
                                 plot_order=["Controls", "AD Patients"]),
        figures.GroupTercileArrows(trimmed_experiment,
                                   plot_order=["Controls", "AD Patients"]),
        figures.GroupTercileCenteringBars(peripheral_experiment,
                                          plot_order=["Controls", "AD Patients"])
    ]

    # manually set the normal distribution limits
    figure_list[1].axes[-1].set_xlim([-300, 300])
    for axis in figure_list[1].axes:
        axis.set_ylim([0, 0.006])
    # add significance annotation
    figure_list[3].annotate_significance(x=[2, 3], label="⁎")

    print(f"Saving figures to disk...")
    for i, figure in enumerate(figure_list):
        if figure is not None:
            figure.name = f"figure_{i}"
            figure.save(output_folder)

    print(f"Saving PIDN list to disk")
    pidn_df.to_csv(f"{output_folder}pidn_list.csv", index=False)


if __name__ == "__main__":
    main()

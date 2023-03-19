# import statements
import numpy as np
import pandas as pd
import figures
from pathlib import Path


class Experiment:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.subjects = {}

    @classmethod
    def from_csv(cls, filepath):
        experiment = cls(pd.read_csv(filepath))
        experiment.update_subjects()
        return experiment

    def copy(self):
        # do a deep copy
        copy = Experiment(self.df.copy())
        copy.subjects = self.subjects.copy()
        return copy

    def update_subjects(self, column_markers=("Group Name", "Subject Name")):
        for group in self.df[column_markers[0]].drop_duplicates():
            self.subjects[group] = [s for s in
                                    self.df[column_markers[1]][self.df[column_markers[0]] == group].drop_duplicates()]

    def to_csv(self, filepath):
        self.df.to_csv(filepath)


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
            for group, subjects in updated_experiment.subjects.items():
                for subject in subjects:
                    subject_trials = updated_experiment.df[(updated_experiment.df[indexing[0]] == group)
                                                           & (updated_experiment.df[indexing[1]] == subject)]

                    # set the original index of each trial
                    subject_trials = subject_trials.assign(**{indexing[2]: np.arange(subject_trials.shape[0])})

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
OUTPUT_PATH = "/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/post-analysis"
COLUMN_MAP = {
    "Group Name": "Group",
    "Subject Name": "Subject",
    "Starting Pitch (Cents)": "InitialPitch",
    "Ending Pitch (Cents)": "EndingPitch",
    "Centering (Cents)": "Centering",
    "Trial Tercile": "Tercile",
    "Trial Index": "Index"
}


def main():
    experiment = Experiment.from_csv(INPUT_PATH)
    print(f"Trials Parsed: {experiment.df.shape[0]}")

    for run in ["auto_trimmed", "manual_exclude"]:
        for std_threshold in [2, 1000]:
            for trim_bundle in ["group", "subject"]:
                # print a horizontal line
                print("-"*80)
                print(f"Using trimming method '{run}' within each {trim_bundle} with std factor {std_threshold}")
                # for auto-trimming
                trimmed_experiment = experiment.copy()
                CenteringAnalysis.Trim.by_subject_trial_count(trimmed_experiment)
                if trim_bundle == "group":
                    CenteringAnalysis.Trim.by_group_initial_pitch_distribution(trimmed_experiment,
                                                                               std_threshold=std_threshold)
                elif trim_bundle == "subject":
                    CenteringAnalysis.Trim.by_subject_initial_pitch_distribution(trimmed_experiment,
                                                                                 std_threshold=std_threshold)

                # for manual trimming
                if run == "manual_exclude":
                    CenteringAnalysis.Trim.by_subject_name(trimmed_experiment, exclude=[("AD Patients", "20160128_1")])

                print(f"Trials trimmed from analysis: {experiment.df.shape[0] - trimmed_experiment.df.shape[0]}")

                CenteringAnalysis.ComputeColumn.tercile(trimmed_experiment)
                CenteringAnalysis.rename_subjects(trimmed_experiment)
                trimmed_experiment.df.rename(columns=COLUMN_MAP, inplace=True)

                peripheral_experiment = CenteringAnalysis.drop_central_trials(trimmed_experiment,
                                                                              inplace=False, indexing="Tercile")

                output_folder = f"{OUTPUT_PATH}/{run}_{std_threshold}_by_{trim_bundle}/"
                # Create the output folder if it doesn't exist already
                Path(output_folder).mkdir(parents=True, exist_ok=True)

                peripheral_experiment.to_csv(f"{output_folder}centering-analysis.csv")
                print(f"Wrote data table to '{output_folder}centering-analysis.csv'")

                for group, subjects in peripheral_experiment.subjects.items():
                    print(f"Including {len(subjects)} subjects in group {group} for this analysis.")

                fig_list = [
                            figures.group_pitch_normal(trimmed_experiment.subjects, trimmed_experiment.df),
                            figures.group_pitch_histogram(trimmed_experiment.subjects, trimmed_experiment.df),
                            figures.group_tercile_centering_bars(peripheral_experiment.subjects,
                                                                 peripheral_experiment.df),
                            figures.group_centering_bars(peripheral_experiment.subjects, peripheral_experiment.df),
                            figures.group_tercile_arrows(trimmed_experiment.subjects, trimmed_experiment.df),
                            figures.group_scatter(trimmed_experiment.subjects, trimmed_experiment.df)
                            ]

                print(f"Saving figures to disk...")
                for figure in fig_list:
                    figure.savefig(f"{output_folder}{figure.name}.png")


if __name__ == "__main__":
    main()

# import statements
import numpy as np
import pandas as pd
import figure_generation as figures


class CenteringAnalysis:
    def __init__(self):
        self.terciles_df = None
        self.output_df = None
        self.trimmed_df = None
        self.subjects = None
        self.input_df = None

    @staticmethod
    def __gen_groups(df: pd.DataFrame):
        groups = [g for g in df["Group Name"].drop_duplicates()]
        subjects = {}

        for group in groups:
            subjects[group] = [s for s in
                               df["Subject Name"][df["Group Name"] == group].drop_duplicates()]

        return subjects

    def import_csv(self, path):
        self.input_df = pd.read_csv(path)
        self.subjects = CenteringAnalysis.__gen_groups(self.input_df)

    # trimming the subjects
    def trim(self, min_trials=None, std_threshold=None, exclude=None):
        # trimmed subjects should be in a separate dataframe
        self.trimmed_df = self.input_df.copy()

        # drop any subject in the manual exclude
        if exclude is not None:
            exclude = []
        for group, subject in exclude:
            print(f"Manually removing {group}/{subject} from the analysis")
            self.trimmed_df.drop(self.trimmed_df[(self.trimmed_df["Group Name"] == group)
                                                 & (self.trimmed_df["Subject Name"] == subject)].index, inplace=True)
            self.subjects = CenteringAnalysis.__gen_groups(self.trimmed_df)

        # remove subjects with not enough trials
        if min_trials is not None:
            for group, subjects in self.subjects.items():
                for subject in subjects:
                    subject_trials = self.trimmed_df[(self.trimmed_df["Group Name"] == group)
                                                     & (self.trimmed_df["Subject Name"] == subject)]
                    if subject_trials.shape[0] < min_trials:
                        print(f"Excl. {group}/{subject}: {subject_trials.shape[0]}/{min_trials}")
                        self.trimmed_df.drop(subject_trials.index, inplace=True)
            self.subjects = CenteringAnalysis.__gen_groups(self.trimmed_df)

        # remove trials with irregular starting pitch 
        if std_threshold is not None:
            for group, subjects in self.subjects.items():
                for subject in subjects:
                    subject_trials = self.trimmed_df[(self.trimmed_df["Group Name"] == group)
                                                     & (self.trimmed_df["Subject Name"] == subject)]

                    subj_mean = np.mean(subject_trials["Starting Pitch (Cents)"])
                    subj_std = np.std(subject_trials["Starting Pitch (Cents)"])

                    outliers = subject_trials[
                        np.abs(subject_trials["Starting Pitch (Cents)"] - subj_mean)
                        > std_threshold * subj_std]

                    self.trimmed_df.drop(outliers.index, inplace=True)

                    print(f"{group}/{subject}: {outliers.shape[0]}/{subject_trials.shape[0]}"
                          " outliers.")

            self.subjects = self.__gen_groups(self.trimmed_df)

        self.trimmed_df.reset_index(inplace=True, drop=True)

    def set_trial_index(self, dataframe=None):
        if dataframe is None:
            dataframe = self.output_df

        for group, subjects in self.subjects.items():
            for subject in subjects:
                subject_trials = dataframe[(dataframe["Group Name"] == group)
                                           & (dataframe["Subject Name"] == subject)]

                # set the original index of each trial
                subject_trials = subject_trials.assign(
                    **{"Trial Index": np.arange(subject_trials.shape[0])})

                # save changes to the original dataframe
                dataframe.update(subject_trials)

    def compute_terciles(self):
        self.terciles_df = self.trimmed_df.copy()
        self.terciles_df["Trial Tercile"] = ""

        for group, subjects in self.subjects.items():
            for subject in subjects:
                subject_trials = self.terciles_df[(self.terciles_df["Group Name"] == group)
                                                  & (self.terciles_df["Subject Name"] == subject)].copy()

                # tercile markers are placed at the 33rd and 66th percentile
                tercile_markers = np.percentile(subject_trials["Starting Pitch (Cents)"],
                                                100 * np.arange(1, 3) / 3)

                subject_trials["Trial Tercile"] = subject_trials.apply(
                    lambda row:
                    "UPPER" if row["Starting Pitch (Cents)"] > tercile_markers[1] else
                    "LOWER" if row["Starting Pitch (Cents)"] < tercile_markers[0]
                    else "CENTRAL", axis=1)

                # save changes to the original dataframe
                self.terciles_df.update(subject_trials)

    def set_output_df(self):
        self.output_df = self.terciles_df.copy()
        # delete all central trials
        self.output_df.drop(self.output_df[
                                (self.output_df["Trial Tercile"] == "CENTRAL")].index, inplace=True)

    def rename_subjects(self, dataframe=None):
        if dataframe is None:
            dataframe = self.output_df

        for group, subjects in self.subjects.items():
            for idx, subject in enumerate(subjects):
                # for each group, rename the subject
                dataframe["Subject Name"].replace(subject,
                                                  (group + str(idx)).replace(" ", ""), inplace=True)

    def rename_columns(self, dataframe=None):
        if dataframe is None:
            dataframe = self.output_df

        dataframe.rename(columns={
            "Group Name": "Group",
            "Subject Name": "Subject",
            "Starting Pitch (Cents)": "InitialPitch",
            "Ending Pitch (Cents)": "EndingPitch",
            "Centering (Cents)": "Centering",
            "Trial Tercile": "Tercile",
            "Trial Index": "Index"
        }, inplace=True)

    def export_csv(self, path):
        self.output_df.to_csv(path, index=False)


# input args
INPUT_PATH = "/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/analyze_audio.csv"
OUTPUT_PATH = "/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/post-analysis/manual/"
MIN_TRIALS = 25


def main():
    analysis_object = CenteringAnalysis()
    analysis_object.import_csv(INPUT_PATH)
    print(f"Trials Parsed: {analysis_object.input_df.shape[0]}")

    analysis_object.trim(min_trials=25, std_threshold=3,
                         exclude=[("AD Patients", "20160128_1")]
                         )
    print(f"Trials trimmed from analysis: "
          f"{analysis_object.input_df.shape[0] - analysis_object.trimmed_df.shape[0]}")

    analysis_object.compute_terciles()

    analysis_object.set_output_df()

    analysis_object.set_trial_index()
    analysis_object.set_trial_index(analysis_object.terciles_df)

    analysis_object.rename_subjects()
    analysis_object.rename_subjects(analysis_object.terciles_df)

    analysis_object.rename_columns()
    analysis_object.rename_columns(analysis_object.terciles_df)

    analysis_object.export_csv(OUTPUT_PATH + "centering-analysis.csv")
    print(f"Wrote data table to {OUTPUT_PATH + 'centering-analysis.csv'}")

    for group, subjects in analysis_object.subjects.items():
        print(f"Including {len(subjects)} subjects in group {group} for this analysis.")

    fig_list = [figures.group_tercile_centering_bars(analysis_object.subjects, analysis_object.output_df),
                figures.group_centering_bars(analysis_object.subjects, analysis_object.output_df),
                figures.group_tercile_arrows(analysis_object.subjects, analysis_object.terciles_df),
                figures.group_scatter(analysis_object.subjects, analysis_object.output_df)]

    print(f"Saving figures to disk...")
    for figure in fig_list:
        figure.savefig(f"{OUTPUT_PATH}{figure.name}.png")


if __name__ == "__main__":
    main()

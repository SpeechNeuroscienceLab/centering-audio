# extra.py
# This file handles many one-off jobs which require basic levels of automation

import pandas as pd
from experiment import Experiment
from scipy.stats import ttest_ind, fisher_exact, mannwhitneyu
from numpy import nan


def compute_demographics(demographics_file_path: str, output_file_path: str, latex: str = None):
    demographics_pd = pd.read_csv(demographics_file_path).dropna(axis=1, how='all')

    # compute the age column
    demographics_pd["DOB"] = pd.to_datetime(pd.to_datetime(demographics_pd["DOB"], format="%m/%d/%y").
                                            dt.strftime("%m/%d/19%y"))
    demographics_pd["Experiment Date"] = pd.to_datetime(demographics_pd["Experiment Date"])
    demographics_pd["Age"] = (demographics_pd["Experiment Date"] - demographics_pd["DOB"]).apply(
        lambda timedelta: timedelta.total_seconds() / (365.2425 * 24 * 3600)
    )

    demographics_summary = pd.DataFrame(columns=["Controls", "AD Patients", "p-value"],
                                        index=["Age", "Female Sex", "White Race", "Education",
                                               "Right Handedness", "CDR", "CDRSOB", "MMSE"])

    # Fill in Age (mean, std)
    # fill in Female Sex (count, percentage)
    for cohort, cohort_abbr in [("Controls", "CONT"), ("AD Patients", "AD")]:
        cohort_df = demographics_pd[demographics_pd["Cohort"] == cohort_abbr]
        print(f"Age range for {cohort}: {min(cohort_df['Age']):.{2}f} - {max(cohort_df['Age']):.{2}f}")
        demographics_summary[cohort]["Age"] = f"${cohort_df['Age'].mean():.{2}f}\\pm{cohort_df['Age'].std():.{2}f}$"

        demographics_summary[cohort]["Female Sex"] = f"{cohort_df['Sex'].value_counts()['Female']}" \
                                                     f" ({100 * cohort_df['Sex'].value_counts()['Female'] / len(cohort_df):.{2}f}\\%)"

        reported_count = cohort_df.shape[0]
        if (cohort_df["Race"] == "DNR").any():
            reported_count = cohort_df[cohort_df['Race'] != 'DNR'].shape[0]
            print(f"{cohort}: {reported_count}/{cohort_df.shape[0]} participants reported race")
        else:
            print(f"{cohort}: all {reported_count} participants reported race.")

        demographics_summary[cohort]["White Race"] = f"{cohort_df['Race'].value_counts()['White']}" \
                                                     f" ({100 * cohort_df['Race'].value_counts()['White'] / reported_count:.{2}f}\\%)"

        demographics_summary[cohort]["Education"] = f"{cohort_df['Education'].median()}" \
                                                    f" ({cohort_df['Education'].quantile(0.25)}-" \
                                                    f"{cohort_df['Education'].quantile(0.75)})"

        demographics_summary[cohort]["Right Handedness"] = f"{cohort_df['Handedness'].value_counts()['RIGHT']}" \
                                                           f" ({100 * cohort_df['Handedness'].value_counts()['RIGHT'] / len(cohort_df):.{2}f}\\%)"

        demographics_summary[cohort]["CDR"] = f"{cohort_df['CDR'].median()}" \
                                              f" ({cohort_df['CDR'].quantile(0.25)}-" \
                                              f"{cohort_df['CDR'].quantile(0.75)})"

        demographics_summary[cohort]["CDRSOB"] = f"{cohort_df['CDRSOB'].median()}" \
                                                 f" ({cohort_df['CDRSOB'].quantile(0.25)}-" \
                                                 f"{cohort_df['CDRSOB'].quantile(0.75)})"

        demographics_summary[cohort]["MMSE"] = f"{cohort_df['MMSE'].median()}" \
                                               f" ({cohort_df['MMSE'].quantile(0.25)}-" \
                                               f"{cohort_df['MMSE'].quantile(0.75)})"

    demographics_summary["p-value"]["Age"] = ttest_ind(demographics_pd[demographics_pd["Cohort"] == "AD"]
                                                       ["Age"],
                                                       demographics_pd[demographics_pd["Cohort"] == "CONT"]
                                                       ["Age"]).pvalue

    for source_column, destination_row in [("Sex", "Female Sex"),
                                           ("Handedness", "Right Handedness")]:
        freq_table = pd.crosstab(demographics_pd["Cohort"], demographics_pd[source_column])
        demographics_summary["p-value"][destination_row] = fisher_exact(freq_table).pvalue

    # special treatment is required for race, since data may not exist
    race_dataset = demographics_pd[demographics_pd["Race"] != "DNR"]
    freq_table = pd.crosstab(race_dataset["Cohort"], race_dataset["Race"])
    demographics_summary["p-value"]["White Race"] = fisher_exact(freq_table).pvalue

    for source_column, destination_row in [("Education", "Education"),
                                           ("MMSE", "MMSE"),
                                           ("CDR", "CDR"),
                                           ("CDRSOB", "CDRSOB")]:
        cohort_dfs = [demographics_pd[demographics_pd["Cohort"] == cohort_abbr][source_column]
                      for cohort_abbr in ["AD", "CONT"]]
        demographics_summary["p-value"][destination_row] = mannwhitneyu(cohort_dfs[0], cohort_dfs[1]).pvalue

    for row in demographics_summary.index:
        demographics_summary["p-value"][row] = "$<0.0001$" if demographics_summary["p-value"][row] < 0.0001 else \
            f"{demographics_summary['p-value'][row]:.{4}f}"

    demographics_summary.to_csv(output_file_path)
    print(f"Wrote demographics data to {output_file_path}")

    if latex is not None:
        demographics_summary.style.to_latex(latex)
        print(f"Wrote LaTeX table to {latex}")

    # "DEMOGRAPHICS LEGEND:"
    # "Values for age are (means, SD)."
    # "Values for education, MMSE, CDR, CDRSOB are (medians, IRQ)."
    # "Unpaired t-test used for Age"
    # "Fisher Exact test for sex, race, and handedness"
    # "Wilcoxon-Mann-Whitney test for education, MMSE, CDR, CDRSOB."


def load_pidn_table(experiment: Experiment):
    PIDN_TABLE_PATH = "/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/pidn-table.xlsx"

    group_aliases = {
        "AD Patients": "AD",
        "Controls": "CONT"
    }

    subject_map = {}

    def subject_to_date_index(subject: str):
        # first, strip away the index
        index = int(subject.split('_')[-1]) - 1 if '_' in subject else 0
        # next, convert to date for the remaining
        date = pd.to_datetime(subject.split('_')[0], format="%Y%m%d")
        return date, index, subject

    for group, subjects in experiment.subjects.items():
        subject_map[group_aliases[group]] = [subject_to_date_index(subject) for subject in subjects]

    pidn_table = pd.read_excel(PIDN_TABLE_PATH)

    pidn_list = pd.DataFrame(columns=["Cohort", "Subject ID", "PIDN"])

    for cohort, subjects in subject_map.items():
        cohort_data = pidn_table[pidn_table["Cohort"] == cohort]

        for subject in subjects:
            matching_date = cohort_data[cohort_data["Date"] == subject[0]]

            match = None
            if matching_date.shape[0] == 0:
                print(f"No PIDN could be found for subject {subject[2]} ({cohort})")
            elif matching_date.shape[0] <= subject[1]:
                print(f"[WARNING] Looking for subject {subject[2]} (SKIPPING)"
                      f"but only found {matching_date.shape[0]} subjects on {subject[0]} ({cohort})")
            else:
                match = int(matching_date.iloc[[subject[1]]]["PIDN"])

            pidn_list = pd.concat([pidn_list,
                                   pd.DataFrame(
                                       {"Cohort": [cohort], "Subject ID": [subject[2]], "PIDN": [match]})],
                                  ignore_index=True)

    return pidn_list

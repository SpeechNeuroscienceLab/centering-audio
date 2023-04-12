# extra.py
# This file handles many one-off jobs which require basic levels of automation

import pandas as pd
from experiment import Experiment


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

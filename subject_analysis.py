from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


class Subject:
    def __init__(self):
        self.name = "N/A"
        self.trials = None
        self.taxis = None


class Cohort:
    def __init__(self, name: str):
        # cohort name is locally identifiable for cohort-level operations
        self.name = name
        self.subjects = []  # contains subjects which in-turn contain trials


class Dataset:
    def __init__(self, path: str, tlv: str):
        self.cohorts = {}

        # load script
        if len(path) > 0 and len(tlv) > 0:
            mat_dict = loadmat(path, simplify_cells=True)

            assert tlv in mat_dict, (f"Top Level Variable {tlv} not found in given .mat file. "
                                     f"Did you mean {mat_dict.keys()}?")
            dataset = mat_dict[tlv]

            for cohort, cohort_data in dataset.items():
                cohort_obj = Cohort(cohort)

                for subject_data in cohort_data:
                    subject_obj = Subject()

                    assert "name" in subject_data, "Subject provided without name."
                    subject_obj.name = subject_data["name"]
                    assert "vonset_frame_taxis" in subject_data, f"Subject {subject_obj.name} does not have variable " \
                                                                 f"'vonset_frame_taxis_in'"
                    subject_obj.taxis = subject_data["vonset_frame_taxis"]

                    assert "vonset_adj_pitch_in" in subject_data, f"Subject {subject_obj.name} does not have variable " \
                                                                  f"vonset_adj_pitch_in"
                    assert "good_trials" in subject_data, f"Subject {subject_obj.name} does not have variable 'good_trials'"
                    # select only good trials
                    subject_obj.trials = subject_data["vonset_adj_pitch_in"][subject_data["good_trials"] == 1, :]

                    cohort_obj.subjects.append(subject_obj)

                assert cohort not in self.cohorts, f"Duplicate cohort: {cohort}"
                self.cohorts[cohort] = cohort_obj

    def __add__(self, other):
        # Note: addition of two datasets doesn't merge within a subject
        merged_dataset = self.copy()
        for cohort in other.cohorts:
            if cohort in merged_dataset.cohorts:
                for subject in other.cohorts[cohort].subjects:
                    # search for subject in merged_dataset.cohorts[cohort].subjects
                    subject_names = [subj.name for subj in merged_dataset.cohorts[cohort].subjects]
                    if subject.name not in subject_names:
                        merged_dataset.cohorts[cohort].subjects.append(subject)
            else:
                merged_dataset.cohorts[cohort] = other.cohorts[cohort]
        return merged_dataset

    def copy(self):
        ds = Dataset("", "")
        ds.cohorts = self.cohorts.copy()
        return ds

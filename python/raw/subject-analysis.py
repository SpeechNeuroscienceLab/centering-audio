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
        mat_dict = loadmat(path, simplify_cells=True)

        assert tlv in mat_dict, f"Top Level Variable {tlv} not found in given .mat file"
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


if __name__ == "__main__":
    print("== Subject-level analysis ==")
    ld = Dataset("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/SD/raw/ld_dataset.mat", "ld_dataset_voice_aligned")

    save_directory = "/Users/anantajit/Documents/Research/UCSF/tables-and-figures/SD/raw/output"

    for cohort_name in ld.cohorts:
        for subject_index, _ in enumerate(ld.cohorts[cohort_name].subjects):
            print(f"Plotting {cohort_name} #{subject_index}")
            subject = ld.cohorts[cohort_name].subjects[subject_index]

            trial_count = subject.trials.shape[0]

            # convert trials to cents
            window_medians = np.median(subject.trials, axis=0)
            subject.trials_cents = 1200 * np.log2(subject.trials/window_medians)

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

            plt.title(f"{subject.name} ({cohort_name} #{subject_index + 1})")
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


    print("DONE.")

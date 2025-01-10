import pandas as pd
import numpy as np

# override print within this scope
log_en = False


def log(string):
    if log_en:
        print(string)


# TODO: make trimming methods functional
def trim_by_subject_name(
        experiment: pd.DataFrame,
        exclude: list,
        indexing=("Group Name", "Subject Name"),
):
    trimmed_experiment = experiment.copy()

    for group, subject in exclude:
        log(f"Manually removing {group}/{subject} from the analysis")
        trimmed_experiment = trimmed_experiment[~((trimmed_experiment[indexing[0]] == group)
                                                  & (trimmed_experiment[indexing[1]] == subject))]

    trimmed_experiment.reset_index(inplace=True, drop=True)

    return trimmed_experiment


def trim_by_subject_trial_count(
        experiment: pd.DataFrame,
        min_trials: int = 25,
        indexing=("Group Name", "Subject Name"),
):
    trimmed_experiment = experiment.copy()

    for group in trimmed_experiment[indexing[0]].unique():
        subjects = trimmed_experiment[trimmed_experiment[indexing[0]] == group][indexing[1]].unique()
        log(f"{group}/{subjects}")
        for subject in subjects:
            subject_trials_df = trimmed_experiment[
                (trimmed_experiment[indexing[0]] == group)
                & (trimmed_experiment[indexing[1]] == subject)
                ]
            if subject_trials_df.shape[0] < min_trials:
                log(
                    f"Excluding {group}/{subject}: {subject_trials_df.shape[0]}/{min_trials}"
                )
                trimmed_experiment.drop(
                    subject_trials_df.index, inplace=True
                )

    trimmed_experiment.reset_index(inplace=True, drop=True)
    return trimmed_experiment


def trim_by_subject_initial_pitch_distribution(
        experiment: pd.DataFrame,
        std_threshold=2,
        indexing=("Group Name", "Subject Name", "Starting Pitch (Cents)"),
):
    trimmed_experiment = experiment.copy()
    for group, subjects in experiment.subjects.items():
        for subject in subjects:
            subject_trials = trimmed_experiment[
                (trimmed_experiment[indexing[0]] == group)
                & (trimmed_experiment[indexing[1]] == subject)
                ]

            subj_mean = np.mean(subject_trials[indexing[2]])
            subj_std = np.std(subject_trials[indexing[2]])

            outliers = subject_trials[
                np.abs(subject_trials[indexing[2]] - subj_mean)
                > std_threshold * subj_std
                ]

            trimmed_experiment.drop(outliers.index, inplace=True)

            log(
                f"{group}/{subject}: {outliers.shape[0]}/{subject_trials.shape[0]} outliers."
            )

    trimmed_experiment.reset_index(inplace=True, drop=True)
    return trimmed_experiment


def trim_by_group_initial_pitch_distribution(
        experiment: pd.DataFrame,
        std_threshold=2,
        indexing=("Group Name", "Starting Pitch (Cents)"),
):
    trimmed_experiment = experiment.copy()

    for group in experiment[indexing[0]].unique():
        group_trials = trimmed_experiment[
            (trimmed_experiment[indexing[0]] == group)
        ]

        subj_mean = np.mean(group_trials[indexing[1]])
        subj_std = np.std(group_trials[indexing[1]])

        outliers = group_trials[
            np.abs(group_trials[indexing[1]] - subj_mean)
            > std_threshold * subj_std
            ]

        trimmed_experiment.drop(outliers.index, inplace=True)

        log(f"{group}: {outliers.shape[0]}/{group_trials.shape[0]} outliers.")

    trimmed_experiment.reset_index(inplace=True, drop=True)
    return trimmed_experiment


def rename_subjects_by_group(
        experiment: pd.DataFrame,
        indexing=("Group Name", "Subject Name"),
):
    verbose = False

    experiment = experiment.copy()

    for group_name, group_data in experiment.groupby(indexing[0]):
        subjects_old = group_data[indexing[1]].unique()
        subjects_new = [str(group_name).replace(" ", "") + str(i) for i in range(1, len(subjects_old) + 1)]
        if verbose:
            for i in range(len(subjects_old)):
                print(f"Renaming {group_name} subjects: {subjects_old[i]} -> {subjects_new[i]}")
        experiment.loc[experiment[indexing[0]] == group_name, indexing[1]] = (
            experiment.loc[experiment[indexing[0]] == group_name, indexing[1]].replace(subjects_old, subjects_new))

    return experiment


def compute_subject_trial_indices(
        experiment: pd.DataFrame,
        indexing=("Group Name", "Subject Name"),
):
    trial_indices = np.empty(experiment.shape[0])

    for group in experiment[indexing[0]].unique():
        subjects = experiment[experiment[indexing[0]] == group][indexing[1]].unique()
        for subject in subjects:
            subject_trials = experiment[(experiment[indexing[0]] == group) & (experiment[indexing[1]] == subject)]
            trial_indices[subject_trials.index] = np.arange(1, subject_trials.shape[0] + 1)
    return trial_indices


def compute_trial_tercile(
        experiment: pd.DataFrame,
        indexing=(
                "Group Name",
                "Subject Name",
                "Starting Pitch (Cents)"
        ),
):
    trial_tercile = np.empty(experiment.shape[0], dtype='<U10')

    for group in experiment[indexing[0]].unique():
        subjects = experiment[experiment[indexing[0]] == group][indexing[1]].unique()
        for subject in subjects:
            subject_data = experiment[(experiment[indexing[0]] == group) & (experiment[indexing[1]] == subject)]
            tercile_markers = np.percentile(subject_data[indexing[2]], 100 * np.arange(1, 3) / 3)
            trial_tercile[subject_data.index] = subject_data.apply(
                lambda row: "UPPER" if row[indexing[2]] > tercile_markers[1]
                else "LOWER" if row[indexing[2]] < tercile_markers[0]
                else "CENTRAL", axis=1)
    return trial_tercile


def cents(
        trials: np.ndarray
):
    window_medians = np.median(trials, axis=0)

    # covers invalid cents conversions
    window_medians[window_medians == 0] = 0.1
    trials[trials <= 0] = 0.1

    trials_cents = 1200 * np.log2(trials / window_medians)
    return trials_cents


def compute_trial_tercile_vector(
        taxis: np.ndarray,
        trials: np.ndarray,
        windows=((0.00, 0.050), (0.150, 0.200)),
        tercile_labels=(0, 1, 2)
):
    trial_count = trials.shape[0]

    trials_cents = cents(trials)

    initial_mask = np.logical_and(taxis >= windows[0][0], taxis <= windows[0][1])
    initial_window_means = np.mean(trials_cents[:, initial_mask], axis=1)
    initial_window_order = initial_window_means.argsort()

    tercile_vector = np.zeros(trial_count)

    for i in range(3):
        trial_indices = initial_window_order[int(i * trial_count / 3): int((i + 1) * trial_count / 3)]

        mask = np.zeros_like(initial_window_order)
        for trial_idx in trial_indices:
            tercile_vector[trial_idx] = tercile_labels[i]

    return tercile_vector

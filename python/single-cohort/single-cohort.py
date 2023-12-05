# imports 
import numpy as np
import csv
import sys



# CONSTANTS 
TRIM_STD = 3

# parse arguments
args = sys.argv

input_path = ""
output_path = ""

verbose = False
printv = lambda text : print(text) if verbose else 0

# for now, automatically do everything
# TODO: arguments control which figures to generate

if ("-v" in args):
    verbose = True
    args.remove("-v")
    printv("Verbose mode enabled")

input_path = args[1]
split_input_path = input_path.split("/")
output_path = "/".join(split_input_path[0:len(split_input_path) - 1]) + "/post-analysis/single/"

printv("Inferred output path: " + output_path)

printv("Initialized settings.")

read_labels = False

printv("Reading input CSV...")

subject_list = {}

trial_data = []

with open(input_path, newline='') as datacsv:
    for row in csv.reader(datacsv):
        if not read_labels:
            read_labels = True
            continue # skip the first row of the CSV
        # otherwise, this is a row of legitimate data 
        subject = (row[0], row[1]) # this way, we keep the subject accessible
        if not(subject in subject_list):
            subject_list[subject] = len(subject_list)

        trial_data.append([subject_list[subject], float(row[2]), float(row[3]), float(row[4]), np.abs(float(row[2])), np.abs(float(row[3])), 0])

# convert to numpy array
trial_data = np.array(trial_data)

printv("Trials loaded: " + str(trial_data.shape))

# clip the data

mean = np.mean(trial_data[:, 2])
std = np.std(trial_data[:, 2])
printv("Cohort mean is " + str(mean) + " with std " + str(std))

deletion_mask = (trial_data[:, 2] > mean + std * TRIM_STD) | (trial_data[:, 2] < mean - std * TRIM_STD)
trial_data = np.delete(trial_data, deletion_mask, axis=0)

printv("Clipped " + str(deletion_mask.sum()) + " trials...")

# analysis of terciles 
printv("Analyzing terciles...")

terciles_data = []

for subject in range(1, len(subject_list)):
    subject_trial_data = trial_data[trial_data[:, 0] == subject, :6]

    labeled_trial_data = np.append(subject_trial_data, np.arange(subject_trial_data.shape[0]).reshape(subject_trial_data.shape[0], 1), 1)
    trials_by_start = labeled_trial_data[labeled_trial_data[:, 2].argsort()]
    tercile_size = int(trials_by_start.shape[0]/3)

    tercile_map = np.zeros([trials_by_start.shape[0], 1])
    tercile_map[0:tercile_size] = -1
    tercile_map[trials_by_start.shape[0] - tercile_size: trials_by_start.shape[0]] = 1

    terciles_trial_data = np.append(trials_by_start, tercile_map, 1)

    # resort data in the original ordering
    terciles_trial_data = terciles_trial_data[terciles_trial_data[:, 6].argsort()]
    terciles_trial_data = np.delete(terciles_trial_data, 6, 1)

    trial_data[trial_data[:, 0] == subject] = terciles_trial_data

    subject_tercile_data = []
    for tercile_idx in range(-1, 2):
        tercile_data = terciles_trial_data[terciles_trial_data[:, 6] == tercile_idx]
        subject_tercile_data.append(np.mean(tercile_data, axis=0))

        terciles_data.append(subject_tercile_data)
printv("Created terciles for " + str(len(terciles_data)) + " subjects")

printv("There are a total of " + len(terciles_data[:, 0]) + " trials.")

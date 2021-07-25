import numpy as np
import csv


# generates a datatable identical to the one by the preprocessor/legacy, but includes the tercile number of each trial
# Group Name | Subject Name | Starting Deviation (Cents) | Ending Deviation (Cents) | Centering (Cents) | Tercile
def run_stats_marked_tercile(group_list, subject_list, trial_data, output_path):
    # initialize output file
    output_path = output_path.rstrip("/")
    with open(output_path + "/run_stats_with_terciles.csv", "w") as file_object:
        output_file = csv.writer(file_object)

        # init labels
        output_file.writerow(["Group Name", "Subject Name", "Starting Deviation (Cents)", "Ending Deviation (Cents)", "Centering (Cents)", "Trial Tercile"])

        for trial in trial_data:
            if trial[7] == 0:
                continue # do not include central trials
            group_idx = int(trial[0])

        #    print(trial)
            group_name = group_list[group_idx]
            subject_name = subject_list[group_idx][int(trial[1])]
            deviations = (float(trial[5]), float(trial[6]))
            centering = float(trial[4])
            tercile = "UPPER" if trial[7] > 0 else "LOWER"

            output_file.writerow([group_name, subject_name, str(deviations[0]), str(deviations[1]), str(centering), tercile])

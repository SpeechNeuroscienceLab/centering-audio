# two arguments REQUIRED, the first should be the input path and the second is the output file path.

import scipy.io as io
import sys
import numpy as np
import csv

raw_data = io.loadmat(sys.argv[1])

raw_data_labels = []

for label in raw_data:
    print("Loaded variable " + str(label) + " successfully.")
    raw_data_labels.append(label)

# for creation of an analysis table, we require:
# Group Name | Subject Name | Starting Pitch (Cents) | Ending Pitch (Cents) | Centering (Cents)

taxis_formant = raw_data["taxis_formant"][0]

windows = ((0, np.argmin(np.abs(taxis_formant - 0.05))), (np.argmin(np.abs(taxis_formant - 0.15)), np.argmin(np.abs(taxis_formant - 0.20))))

print("Computed Windows: " + str(windows))

phase_list = ["post", "pre"]

group_list = ["control", "lido"]

formant_list = ["F1"]

trial_length = 763

header = [["Group Name","Subject Name","Starting Pitch (Cents)","Ending Pitch (Cents)","Centering (Cents)"]]

table_comb = {}

for group in group_list:
    for phase in phase_list:
        for formant in formant_list:
            table_data = []
        
            data_target = group + "_" + phase + "_allraw_" + formant + "_zero"
            print("Loading " + data_target)
            
            data = raw_data[data_target][0]
            
            for sidx, subject in enumerate(data):
                subject_name = group + str(sidx)
                
                subject_data = np.ones((len(data[sidx]), trial_length))

                for tidx, trial in enumerate(data[sidx]):
                    subject_data[tidx, :] = data[sidx][tidx]
                
                window_vals = (subject_data[:, windows[0][0]:windows[0][1]], subject_data[:, windows[1][0]:windows[1][1]])
                
                window_vals = (window_vals[0][~np.isnan(window_vals[0]).any(axis=1)], window_vals[1][~np.isnan(window_vals[1]).any(axis=1)])
                
                
                window_means = (np.nanmean(window_vals[0], axis=1), np.nanmean(window_vals[1], axis=1))
                
                window_medians = (np.nanmedian(window_means[0]), np.nanmedian(window_means[1]))
                
                
                window_vals_cents = (1200 * np.log(window_means[0]/window_medians[0])/np.log(2),1200 * np.log(window_means[1]/window_medians[1])/np.log(2))
                
                centering_values = np.abs(window_vals_cents[0]) - np.abs(window_vals_cents[1])
                
                for i in range(len(centering_values)):
                    table_data.append([group + phase, subject_name, window_vals_cents[0][i], window_vals_cents[1][i], centering_values[i]])
                
                table_comb[(group, phase)] = table_data

# build tables
tables = [table_comb[("control", "pre")] + table_comb[("control", "post")], table_comb[("lido", "pre")] + table_comb[("lido", "post")], table_comb[("control", "pre")] + table_comb[("lido", "pre")],table_comb[("control", "post")] + table_comb[("lido", "post")],]
table_names = ["control_pre_post", "lido_pre_post", "pre_control_lido", "post_control_lido"]


for i in range(len(tables)):
    path = sys.argv[2] + "_" + table_names[i] + ".csv"
    with open(path, "w") as outputfile:
        wr = csv.writer(outputfile)
        wr.writerows(tables[i])

        print("Wrote " + str(len(tables[i])) + " rows to " + path)

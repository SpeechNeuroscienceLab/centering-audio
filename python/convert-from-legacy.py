# This script takes a CSV that was exported from the MATLAB version of the centering-audio
# and converts into a version that can be used for visualization. We could just modify the
# existing matlab, but the current structure of the MATLAB code is still not as modular as it
# potentially could be. Since the future is python in any case, this script is used to update
# the datatable with the new definition of centering, as well as remove the unnecessary parts
# of the datatable. 

# input datatable with columns:
# Subject name, Trial Centering Value, Trial Deviation Value, Deviation Sign, Centering over Deviation Ratio, Group Name

# run_stats table
# Group Name, Subject Name, Starting Deviation, Ending Deviation, Centering Value


# import statements
import numpy as np
import csv

def init_export_writers(exports, output_folder):
    file_writers = dict()
    for filename in exports:
        file_writers[filename] = csv.writer(open(output_folder + "/" +  filename + ".csv", "w"))
    return file_writers

def export_headings(exports, file_writers):
    line_data = dict()
    line_data["group_name"] = "Group Name"
    line_data["subject_name"] = "Subject Name"
    line_data["trial_deviations_cents"] = ("Starting Deviation (Cents)", "Ending Deviation (Cents)")
    line_data["trial_signed_deviations_cents"] = ("Starting Pitch (Cents)", "Ending Pitch (Cents)")
    line_data["trial_centering_cents"] = "Centering (Cents)"
    export_line_data(exports, line_data, file_writers)

def export_line_data(exports, line_data, file_writers):
    if "run_stats" in exports:
        # export the run_stats export
        file_writers["run_stats"].writerow([ \
                line_data["group_name"], \
                line_data["subject_name"], \
                str(line_data["trial_deviations_cents"][0]), \
                str(line_data["trial_deviations_cents"][1]), \
                str(line_data["trial_centering_cents"])])
    if "analyze_audio" in exports:
        file_writers["analyze_audio"].writerow([ \
                line_data["group_name"], \
                line_data["subject_name"], \
                str(line_data["trial_signed_deviations_cents"][0]), \
                str(line_data["trial_signed_deviations_cents"][1]), \
                str(line_data["trial_centering_cents"])])

# enable debugging
# import pdb
# pdb.set_trace()

# TODO provide flexible file interaction using flags and arguments, or input (or both)
input_filepath = "/storage/Organizations/UCSF/tables-and-figures/SD/sddatatable.csv"
output_filepath = "/storage/Organizations/UCSF/tables-and-figures/SD/"


# cleanup filepath for output
output_filepath = output_filepath.rstrip("/")

# TODO flags from input 
exports = ["run_stats", "analyze_audio"]

# initialize files
export_writers = init_export_writers(exports, output_filepath)

initialized_headings = False

# load the input file
with open(input_filepath, newline='') as datacsv:
    # for each line
    # the first line should be labels. 
    # TODO Verify that the labels match the expected input, otherwise issue a warning
    
    for row in csv.reader(datacsv):
        if not initialized_headings:
            export_headings(exports, export_writers)
            initialized_headings = True
            continue
        # Assumes that the first row was labels
        rowdata = dict()
        rowdata["group_name"] = row[5]
        rowdata["subject_name"] = row[0]
        pitches = (float(row[2]) if row[3] == "POSITIVE" else -1 * float(row[2]), -1 * float(row[2]) + float(row[1]) if row[3] == "NEGATIVE" else float(row[2]) - float(row[1]))

        # now we can recalculate the centering
        centering = np.abs(pitches[0]) - np.abs(pitches[1])
        
        rowdata["trial_signed_deviations_cents"] = pitches
        rowdata["trial_deviations_cents"] = np.abs(pitches)
        rowdata["trial_centering_cents"] = centering


        export_line_data(exports, rowdata, export_writers)

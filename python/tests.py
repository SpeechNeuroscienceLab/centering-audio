from scipy.stats import norm, kstest
import numpy as np
import matplotlib.pyplot as plt

test_output = []

def save_results(output_path):
    output_path = output_path.rstrip("/")
    with open(output_path + "/test_results.txt", "w") as output_file:

        output_file.write("###################### TEST  RESULTS ########################\n")
        for test in test_output:
            output_file.write(test + "\n")

def print_test_results():
    print("###################### TEST  RESULTS ########################")
    for test in test_output:
        print(test + "\n")

########################### TEST OF NORMALITY ############################
# Test of Normality will be run for the starting deviations as well as the 
# ending deviations. 

def test_of_normality(group_list, trial_data):
    # runs test of normality on data for each group
    for group_idx in range(0, len(group_list)):
         group_name = group_list[group_idx]
         group_data = trial_data[trial_data[:, 0] == group_idx, :]
         starting_deviations = group_data[:, 2]

         mean = np.mean(starting_deviations) 
         std = np.std(starting_deviations)

         result = kstest(starting_deviations, 'norm', args=(mean, std))
         # the arguments mean and std attempt to compare the starting deviations with
         # a normal distribution with a matching mean and standard deviation. This works
         # flawlessly for perfectly normal random distributions, and fails for uniform. 

         test_output.append("Test of Normality for " + group_name + " Starting Deviations:\n" + str(result) + "\n")

         ending_deviations = group_data[:, 3]

         mean = np.mean(ending_deviations) 
         std = np.std(ending_deviations)

         result = kstest(ending_deviations, 'norm', args=(mean, std))
         # the arguments mean and std attempt to compare the starting deviations with
         # a normal distribution with a matching mean and standard deviation. This works
         # flawlessly for perfectly normal random distributions, and fails for uniform. 

         test_output.append("Test of Normality for " + group_name + " Ending Deviations:\n" + str(result) + "\n")


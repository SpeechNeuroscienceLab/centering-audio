import scipy
import numpy as np

test_output = []

def print_test_results():
    print("###################### TEST  RESULTS ########################")
    for test in test_output:
        print(test)

########################### TEST OF NORMALITY ############################
# Test of Normality will be run for the starting deviations as well as the 
# ending deviations. 
def test_of_normality(group_list, trial_data):
    # runs test of normality on data for each group
    for group_idx in range(0, len(group_list)):
        group_name = group_list[group_idx]




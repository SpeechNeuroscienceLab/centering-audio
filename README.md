# Centering Analysis Scripts

## File Structure

`analysis.py` contains analysis functions (for example, trimming, centering and tercile computation). `figure.py` contains all plotting functions. `pipeline.py` contains a pipeline function definition, which is used during analysis to chain together multiple function calls so many analysis steps can be applied to the same data table in a single line. `subject_analysis.py` contains other useful data structures. For each "experiment," there should be a different Python file (examples for AD and LD are in the repository). This is the "top level" file that should be run by the Python interpreter.

## Running Analysis

Running `compute_xx_centering.py` runs centering analysis. The current implementations use cached results for the analysis component (it would be very useful if this same idea could be applied to the plotting code too). Caching can be disabled by using the `FORCE_ANALYSIS` variable to `True` (in existing implementations).

## Input Formats

The current script design enables injecting the cached state directly. For example, the AD dataset was provided as a CSV (intermediate format) with precomputed pitch centering data in cents. On the other hand, the LD dataset was imported directly from MATLAB data structures.

To import data from MATLAB, save the data into a single MATLAB struct, which itself is saved into a `.mat` file. Then, pass this into a `Dataset` constructor (`subject_analysis.py`). 

# Centering Analysis Scripts

## File Structure

`analysis.py` contains analysis functions (for example, trimming, centering and tercile computation). `figure.py` contains all plotting functions. `pipeline.py` contains a pipeline function definition, which is used during analysis to chain together multiple function calls so many analysis steps can be applied to the same data table in a single line. `subject_analysis.py` contains other useful data structures. For each "experiment," there should be a different Python file (examples for AD and LD are in the repository). This is the "top level" file that should be run by the Python interpreter.

## Running Analysis

Running `compute_xx_centering.py` runs centering analysis. The current implementations use cached results for the analysis component (it would be very useful if this same idea could be applied to the plotting code too). Caching can be disabled by using the `FORCE_ANALYSIS` variable to `True` (in existing implementations).

### Input Format

The current script design enables injecting the cached state directly. For example, the AD dataset was provided as a CSV (intermediate format) with precomputed pitch centering data in cents. On the other hand, the LD dataset was imported directly from MATLAB data structures.

To import data from MATLAB, save the data into a single MATLAB struct, which itself is saved into a `.mat` file. Then, pass this into a `Dataset` constructor (`subject_analysis.py`). The `.mat` must have the following format: 

```
Top Level Variable {}
├─ Cohort A[]
│  ├─ Subject 1 {}
│  │  ├─ name
│  │  ├─ vonset_adj_pitch_in
│  │  ├─ vonset_frame_taxis
│  │  ├─ good_trials
│  ├─ Subject 2 {}
├─ Cohort B[]
│  ├─ ...
├─ ...
```

The group annotation `{}` denotes a MATLAB structure and `[]` denotes a (cell) array. The subject field `name` is a string, and `vonset_adj_pitch_in`, `vonset_frame_taxis` and `good_trials` are all vector arrays, consistent with the variable naming scheme from the MATLAB analysis scripts.

### Importing Data

To import data from MATLAB into Python, use the `Dataset` constructor in the `subject_analysis` class.

```python
from subject_analysis import Dataset
dataset = Dataset(path_to_dataset, top_level_variable_name)
```

This returns a `Dataset` object, with each cohort falling into a `Cohort` object stored in dict `dataset.cohorts`. Each cohort contains a list of `Subject` objects in a list `cohort.subjects` which each contain a `name`, `trials` matrix and time axis vector `taxis`. Note that no unit conversions are performed during this analysis - if values were originally in hz, then the pitch values in the `trials` matrix will also be in hz.

### Intermediate Data Conversions

The majority of data analysis is done using CSV data. In the existing codebase, the columns are named "Group Name", "Subject Name", "Starting Pitch (Cents)", "Ending Pitch (Cents)" and "Centering (Cents)", though everything is configurable through parameterization in function calls. In `compute_ld_centering.py`, the centering table is conditionally generated (if not cached, that is), and the centering table is maintained in a pandas dataframe with the same column names. This datatable (or subsets of the centering table, in cases of trimming) are passed into figure functions.

### Figure Generation

Figures are plotted using functions in `figure.py`. Each function generates one figure, and generally takes the following format: 

```python
import figure
figure_object = plt.figure()
figure_object.add_axes((...))
figure.plotting_function(input_dataframe, figure_object, plot_settings)
figure_object.savefig(...)
plt.close()
```

`plot_settings` dictionary contains a series of values which each plot depends on. I overrode these values using the Python `|` operation with my overrides. 

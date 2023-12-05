# centering-audio

A library of Matlab scripts that are used for analyzing the audio components of speech centering.

## Functions
`experiment_pitch_centering(groups)` - preprocessor for pitch centering data that returns a summary datatable for multiple groups, given argument `groups`:
- `groups` ({struct)}: cell array of structures of form `GROUP`
- `GROUP` (struct): structure with properties
	- `GROUP.name` (string): name of group (ex. Patients, Controls)
	- `GROUP.path` (string): path to folder that contains all subjects in group
	- `GROUP.subjects` ({string}): cell array of strings that contains the names of each subject

`production_figures(datatable)` - generates visuals based on datatable generated by `experiment_pitch_centering` or `experiment_formant_centering`.
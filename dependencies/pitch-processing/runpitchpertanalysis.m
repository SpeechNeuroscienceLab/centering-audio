clear all; close all;
speak_consolidate_audiodir = 'speak_consolidate_audiodir';

isubj = 1;

subj_info(isubj) = get_subj_info_dirs(speak_consolidate_audiodir);
[vods,pitchlimits, perttrial_pre, perttrial_dur, pertclass_pitch_types, yes_overwrite] = get_init_info(subj_info);
subj_info(isubj).pitchlimits = pitchlimits;

 runpitchpertanalysis_core
 
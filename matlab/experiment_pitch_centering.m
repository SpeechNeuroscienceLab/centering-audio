% Anantajit Subrahmanya
% September 20 2020
%
% Preprocess all of the subjects in 
% cell array of "subject groups". 
% Returns a datatable that combines all of the subjects into a single cell
% array that can be pasted into Excel.
% 
% PARAM groups - (cell vector of group structures)
% a group is a structure that provides information about a set of subjects. 
% A group GROUP should have the following properties:
%   GROUP.PATH (string) - path of the group that contains each of the
%   subjecs as a subfolder. For example, if the group path was /data/group,
%   and subjects subj1 and subj2 were part of the group, then the subjects
%   should be stored in /data/group/subj1 and in /data/group/subj2.
%   GROUP.NAME (string) - name of the group. This is only used for the
%   "group name" parameter in the datatable that is generated by this
%   script
%   GROUP.SUBJECTS (cell vector of strings) - each subject that is to be
%   processed should be listed here. For example, {'subj1', 'subj2'}
%   Notice that all the subjects MUST be valid, as errors due to a subject
%   being malformated will not be caught elegantly. 
%
% This script assumes that the data is of the same format as each of the
% following *functional* group folders
% /data/research_meg10/hardik/SD-pitch/Patients or
% /data/research_meg10/hardik/SD-pitch/Controls or
% /data/bil_mb8/houde_lab/Hardik/AD_Processed_Data/Patients or 
% /data/bil_mb8/houde_lab/Hardik/AD_Processed_Data/Controls
%
% Example usage of this function: 
% datatable_of_g1_g2 = experiment_pitch_centering({group1, group2});
%
% RETURN datatable - Returns a datatable that combines all of the trials
% from the groups that are input. 
% Datatable form is as follows: 
% Subject Name | Centering Value | Deviation Value | Deviation Sign | Centering over Deviation Ratio | Group Name
%

function datatable = experiment_pitch_centering(groups)

number_of_groups = max(size(groups));

init_path = pwd;

datatable = {'Subject Name', 'Centering Value', 'Deviation Value', 'Deviation Sign', 'Centering over Deviation Ratio', 'Group Name'};

for group_idx = (1:number_of_groups)
    group = groups{group_idx};
    
    cd(group.path);
    disp(strcat('STARTING WORK ON: ', group.name));
   
    % for each of the *whitelisted* subjects only.
    number_of_subjects = max(size(group.subjects));
    
    for subject_idx = (1:number_of_subjects)
       subject_id = group.subjects{subject_idx};
       
       cd(subject_id);
       
       subject_datatable = subject_pitch_centering(subject_id, group.name);
       
       datatable = append_rows(datatable, subject_datatable);
       
       cd('../');
    end
    
end

% restore the original path, for convenience
cd(init_path);

end


% Generates the datatable (same format) for a single subject. 
function [datatable] = subject_pitch_centering(subject_name, group_name)

disp('STARTING NEW SUBJECT:');
disp(subject_name);

% dependencies that are necessary for the script to run. TODO: set up
% automatic linking using GitHub Packages
addpath('/data/research_meg9/anantajit/pitch/dependencies/wave-viewer');
addpath('/data/research_meg9/anantajit/pitch/dependencies/pitch-processing');

% PRECONDITION: assume that all of the subjects are valid.
cd('speak_consolidate_audiodir');

load('pitch_trials.mat');
load('parsed_pitch_trials_test.mat');

cd('../');

%all the necessary *files* have been loaded.

number_of_trials = ntrials;
frames_per_trial = nframes;
time_axis = taxis;

number_of_good_trials = sum(good_trials);

subject_trials_raw = NaN(number_of_trials, frames_per_trial);

trial_index = 1;

run_list = folder_filter(list_of_folders(), {'expr'}, {});

% foreach run...

for run_number = (1:size(run_list, 1))
    cd(run_list{run_number, 1});
    
    cd('speak');
    block_list = folder_filter(list_of_folders(), {'block'}, {'consolidate'});
    
    % foreach speak block
    for block_number = (0:size(block_list, 1)-1)
        cd(block_list{block_number+1, 1});
        
        % get the trials within the block
        voice_onset_binary_array = get_vec_hist6('voice_onset_detect', 2);
        voice_onset_binary_array = voice_onset_binary_array.data;
        
        raw_audio_array = get_vec_hist6('inbuffer', 3);
        load('pitch_trials.mat');
        trials_per_block = ntrials;
        % foreach trial
        for trial_number = (1:trials_per_block)
            % if bad, ignore trial
            if(good_trials(trial_index, 1) == 0)
                trial_index = trial_index + 1;
                continue;
            end
            % otherwise copy the contents of the trial into
            % subject_trials_raw
            subject_trials_raw(trial_index, :) = pitch_v2_vec_hist6(raw_audio_array, trial_number);
            subject_trials_raw(trial_index, ~voice_onset_binary_array(trial_number, :)) = nan;
            trial_index = trial_index + 1;
        end
        
        cd('../');
    end
    cd('../');
    
    
    cd('../');
end


% all raw trials (hz) have been loaded

% windows are in milliseconds (0 to 50ms, 150 to 200ms)
% TODO: config file that contains all constants
centering_windows = [0 50; 150 200];

subject_trial_windows_hz = NaN(number_of_trials, 2);

for trial_idx = (1:number_of_trials)
    trial = subject_trials_raw(trial_idx, :);
    % find first nonnan value... this will be the first "voice onset" value
    % since the data has already been trimmed to voice onset. 
    voice_onset = find(~isnan(trial), 1);
    if(isempty(voice_onset))
        % bad trial
        continue;
    end
    
    % voice onset has been found. record the time
    adjusted_centering_windows = centering_windows/1000 + taxis_frame(voice_onset);
    for window_idx = (1:2)
        window_start_idx = find(taxis_frame >= adjusted_centering_windows(window_idx, 1), 1);
        window_end_idx = find(taxis_frame >= adjusted_centering_windows(window_idx, 2), 1);
        
        subject_trial_windows_hz(trial_idx, window_idx) = nanmean(trial(window_start_idx:window_end_idx));
    end
    
end


% convert to cents.

baseline = zeros(2, 1);
baseline(1:2, 1) = nanmedian(subject_trial_windows_hz(:, (1:2)));

subject_trial_windows_cents = zeros(number_of_trials, 2);
for window_idx = (1:2)
    subject_trial_windows_cents(:, window_idx) = 1200 * log2(subject_trial_windows_hz(:, window_idx)/baseline(window_idx,1));
end

% clean up... any weird cases (for ex. trial is too short, <200ms) are deleted 
subject_trial_windows_cents(isnan(subject_trial_windows_cents(:, 1)) | isnan(subject_trial_windows_cents(:, 2)), :) = [];

% calculate the deviations
subject_deviations_cents = abs(subject_trial_windows_cents(:, 1));

% calculate diff for centering
subject_centering_cents = subject_trial_windows_cents(:, 2) - subject_trial_windows_cents(:, 1);

% flip... st always positive centering if going towards center
subject_centering_cents(subject_trial_windows_cents(:, 1) > 0) = -1 * subject_centering_cents(subject_trial_windows_cents(:, 1) > 0);

% centering and deviation are complete.

subject_centering_deviation_ratio = subject_centering_cents./subject_deviations_cents;

datatable = {};

% append the trials to the subject's datatable
for trial_idx = (1:size(subject_centering_cents, 1))
    centering = subject_centering_cents(trial_idx, 1);
    deviation = subject_deviations_cents(trial_idx, 1);
    deviation_signed = subject_trial_windows_cents(trial_idx, 1);
    if(deviation_signed < 0)
        deviation_sign = {'NEGATIVE'};
    else
        if(deviation_signed > 0)
            deviation_sign = {'POSITIVE'};
        else
            deviation_sign = {'ZERO'};
        end
    end
    ratio = subject_centering_deviation_ratio(trial_idx, 1);
    
    data = {subject_name, centering, deviation, deviation_sign, ratio, group_name};
    datatable = append_rows(data, datatable);
    
end

number_of_runs = 0;

end

% with a bunch of words that must appear (filter) and a bunch of words that
% should never appear (ignore), find all strings (folders) that satisfy
% this requirement.
function [valid_folders] = folder_filter(folders, filters, ignore)

valid_folders = {};

for i = (1:size(folders, 1))
    test_folder = folders{i};
    good_folder = 1;
    for j = (1:max(size(ignore)))
       filter = ignore{j};
       if(~isempty(strfind(test_folder, filter)))
          good_folder = 0;
          break;
       end
    end
    for j = (1:max(size(filters)))
        % foreach filter
        filter = filters{j};
        if(isempty(strfind(test_folder, filter)))
            good_folder = 0;
            break;
        end
    end
    if(good_folder)
       valid_folders{size(valid_folders, 1) + 1, 1} = test_folder; 
    end
end

end

% Generates a list of all subdirectories only. 
function [folder_list] = list_of_folders()
    % Get a list of all files and folders in this folder.
files = dir;
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subfolders = files(dirFlags);
% Print folder names to command window.
folder_list = cell(max(size(subfolders)), 1);
for i = (1:size(folder_list, 1))
   folder_list{i} = subfolders(i).name; 
end
end

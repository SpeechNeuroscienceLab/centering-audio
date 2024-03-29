% Anantajit Subrahmanya
% September 25 2020
%
% Standalone utility to preprocess matlab array of raw subject data into formant centering datatable. 
% Returns a datatable that combines all of the subjects into a single cell
% array that can be pasted into Excel.
% 
% PARAM groups - (cell vector of group structures)
% a group is a structure that provides information about a set of subjects. 
% A group GROUP should have the following properties:
%   GROUP.NAME (string) - name of the group. This is only used for the
%   "group name" parameter in the datatable that is generated by this
%   script
%   GROUP.SUBJECT_NAMES (cell vector of strings) - each subject that is to be
%   processed should be listed here. For example, {'subj1', 'subj2'}
%   Notice that all the subjects MUST be valid, as errors due to a subject
%   being malformated will not be caught elegantly. 
%   GROUP.SUBJECT_DATA (cell array of double arrays) - indicies should
%   match those of GROUP.SUBJECT_NAMES. In other words, the first subject
%   in GROUP.SUBJECT_NAMES will have raw data in the first index of
%   SUBJECT_DATA. The row number in the double array matches the trial
%   number; so each trial takes up one row. 
%   GROUP.TAXIS (double vector) - shows the conversion from frame (column
%   number) to time in seconds. 
%   GROUP.BLACKLISTS (cell array of strings) - contains a cell array of
%   subjects that you do not wish to include in the final datatable...
%   useful for quickly removing outliers.
%
% Example usage of this function: 
% datatable_of_g1_g2 = experiment_formant_centering({group1, group2});
%
% RETURN datatable - Returns a datatable that combines all of the trials
% from the groups that are input. 
% Datatable form is as follows: 
% Subject Name | Centering Value | Deviation Value | Deviation Sign | Centering over Deviation Ratio | Group Name
%
function [datatable, debug] = experiment_formant_centering(groups, export_path)

% this is a modified version of experiment_pitch_centering, but this does
% not cover any of the graphics at this stage.should run much faster

debug = {};
datatable = {'Subject Name', 'Centering Value', 'Deviation Value', 'Deviation Sign', 'Centering over Deviation Ratio', 'Group Name'};


% Constants that will be used for processing

% centering window times are in milliseconds.
precentering_window = [0 50];
postcentering_window = [150 200];

number_of_groups = size(groups, 2);

fprintf('NUMBER OF GROUPS FOUND: %d\n', number_of_groups);

% there should be at least one group

for group_idx = (1:number_of_groups)
    group = groups{1, group_idx};
    fprintf('WORKING ON: %s\n', group.name);
    taxis = group.taxis;
    % using the taxis, find the frames for the precentering window.
    precentering_window_frames = [find(taxis > precentering_window(1)/1000, 1)  find(taxis > precentering_window(2)/1000, 1)];
    postcentering_window_frames = [find(taxis > postcentering_window(1)/1000, 1)  find(taxis > postcentering_window(2)/1000, 1)];
    
    number_of_subjects = size(group.subject_names, 2);
    
    for subject_idx = (1:number_of_subjects)
        subject_name = group.subject_names{1, subject_idx};
        subject_trials = group.subject_data{1, subject_idx};
        
        
        if(strarr_contains(group.blacklists, subject_name))
            % this is a part of the blacklist...  should skip it
            continue;
        end
        % not a blacklisted subject,  can process it.
        
        number_of_trials = size(subject_trials, 1);
        
        pitch_position_hz = zeros(number_of_trials, 2);
        
        for trial_idx = (1:number_of_trials)
            trial = subject_trials(trial_idx, :);
            %  cannot calculate the centering prior to normalizing the
            % data into cents. For this,  need to iterate through all of
            % the trials first.
            
            %  will first generate a list of the "start point" and the
            % "end point"
            pitch_position_hz(trial_idx, 1) = nanmean(trial(1, precentering_window_frames(1):precentering_window_frames(2)));
            pitch_position_hz(trial_idx, 2) = nanmean(trial(1, postcentering_window_frames(1):postcentering_window_frames(2)));
        end
        
        %  collected all of the values that we need from the trial.
        %  find the "center"
        
        center = [nanmedian(pitch_position_hz(:, 1)) nanmedian(pitch_position_hz(:, 2))];
        
        % by using the center, can adjust the rest of the values into
        % cents.
        pitch_position_cents = zeros(number_of_trials, 2);
        
        pitch_position_cents(:, 1) = 1200 * log2(pitch_position_hz(:, 1)/center(1, 1));
        pitch_position_cents(:, 2) = 1200 * log2(pitch_position_hz(:, 2)/center(1, 2));
        
        % conversion to cents is complete!
        
        % can now calculate the centering of each of these rows. The
        % centering will be placed in its own column.
        centering = pitch_position_cents(:, 2) - pitch_position_cents(:, 1);
        
        % note that in cases where the the starting point is positive, 
        % need to flip our centering value because it is supposed to be
        % positive when moving towards the center
        positive_deviations = pitch_position_cents(:, 1) > 0;
        
        centering(positive_deviations) = -1 * centering(positive_deviations);
        
        %  All of the required components have already been calculated.
        
        for trial_idx = (1:number_of_trials)
            if(positive_deviations(trial_idx))
                deviation_sign = {'POSITIVE'};
            else
                deviation_sign = {'NEGATIVE'};
            end
            datatable = append_rows(datatable, {subject_name, centering(trial_idx), abs(pitch_position_cents(trial_idx, 1)), deviation_sign,  centering(trial_idx)/abs(pitch_position_cents(trial_idx, 1)), group.name});
        end
        
    end
end

return;

end

% Quick little utility that checks of a string array contains a given
% string. Basically simple array searching. 
function [index] = strarr_contains(strarr, str)

for index = (1: length(strarr))
   if(strcmp(strarr{index}, str))
      return; 
   end
end
index = 0;
return;
end
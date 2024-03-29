% Anantajit Subrahmanya
% September 20 2020
% Draws figures (misc) 
% PARAM datatable - datatable generated by experiment_pitch_centering
% RETURN debug - misc debugging data... 
% RETURN output - set of datatables
% TODO: clean up scripts to make more modular. 

function [debug, output] = production_figures(datatable)

% start of code

debug = {};
output = {};

% GENERAL data parsing and sorting.

data = {};

% current support for this script is for two groups only.
datatable_copy = datatable;
%clip the first line, don't modify datatable original
datatable_copy(1,:) = [];
% sort by the two groups
while(~isempty(datatable_copy))
    % sort into the appropriate group and then into the appropriate subject
    trial = datatable_copy(1, :);
    % delete this trial from the datatable
    datatable_copy(1, :) = [];
    % find the group name
    group_name = trial(1, 6);
    % check if this group has already been created or not
    group_idx = 0;
    for i = (1:size(data))
        if(isequal(data{i}.name, group_name))
            group_idx = i;
            break;
        end
    end
    
    if(~group_idx)
        % create it manually
        group_idx = size(data, 1) + 1;
        data{group_idx, 1} = {};
        data{group_idx}.name = group_name;
        data{group_idx}.subjects = {};
    end
    
    subject_name = trial(1,1);
    subject_idx = 0;
    for i = (1:size(data{group_idx, 1}.subjects))
        if(isequal(data{group_idx, 1}.subjects{i}.name, subject_name))
            subject_idx = i;
            break;
        end
    end
    
    if(~subject_idx)
        subject_idx = size(data{group_idx, 1}.subjects, 1) + 1;
        data{group_idx, 1}.subjects{subject_idx, 1} = {};
        data{group_idx, 1}.subjects{subject_idx, 1}.name = subject_name;
        data{group_idx, 1}.subjects{subject_idx, 1}.trials = {};
    end
    
    % append this trial to the list of the subject's trial. 
    trial_datablock = trial(1, 2:5);
    % add "effective" deviation in position 5. This is the "real" deviation
    % value
    deviation_effective = trial{1, 3}(1,1);
    if(isequal(trial{1, 4}{1,1}, 'NEGATIVE'))
        deviation_effective = deviation_effective * (-1);
    end
    trial_datablock(1, 5) = {deviation_effective};
    data{group_idx, 1}.subjects{subject_idx, 1}.trials = append_rows(trial_datablock, data{group_idx, 1}.subjects{subject_idx, 1}.trials);
end

% note that at this point, we still need to put everything back in terms of
% terciles. We will do that tercile calculation now.

for group_idx = (1:size(data, 1))
    group = data{group_idx, 1};
    for subject_idx = (1:size(group.subjects, 1))
        subject = group.subjects{subject_idx, 1};
        % take all of this subject's values and sort them by the "effective deviation
        % note that this is only a local change, not a global change that
        % applies to our dataset.
        subject.trials = sortrows(subject.trials, 5);
        tercile_size = floor(size(subject.trials, 1)/3);
        tercile_divider = [subject.trials{1 + tercile_size, 5}, subject.trials{size(subject.trials, 1) - tercile_size, 5}];
        
        for trial_idx = (1:size(subject.trials, 1))
            % for each trial, figure out if it is an upper or lower or middle tercile trial.
            if(data{group_idx, 1}.subjects{subject_idx, 1}.trials{trial_idx, 5} >= tercile_divider(2))
                tercile_number = 3;
            else if(data{group_idx, 1}.subjects{subject_idx, 1}.trials{trial_idx, 5} < tercile_divider(1))
                    tercile_number = 1;
                else
                    tercile_number = 2;
                end
            end
            data{group_idx, 1}.subjects{subject_idx, 1}.trials{trial_idx, 5} = tercile_number;
        end
    end
end


% Datatable 1 and 2 - plotting data
% Subject | Mean Centering | Mean Deviation
datatables = {};

for group_idx = (1:size(data, 1))
    group = data{group_idx, 1};
    
    tercile_means = cell(3, 1);
    
    for subject_idx = (1:size(group.subjects, 1))
        subject = group.subjects{subject_idx, 1};
        subject_name = subject.name;
        
        trials_by_tercile = cell(3,1);
        for trial_idx = (1:size(subject.trials, 1))
            trial = subject.trials(trial_idx, :);
            tercile_number = trial{1,5};
            
            % break this into the trials by tercile for each group.
            trials_by_tercile{tercile_number, 1} = append_rows(trial, trials_by_tercile{tercile_number, 1});
        end
        
        for tercile_num = ([1 3])
            mean_deviation = table_column_mean(trials_by_tercile{tercile_num, 1}, 2);
            mean_centering = table_column_mean(trials_by_tercile{tercile_num, 1}, 1);
            data_value = {subject_name mean_centering mean_deviation};
            tercile_means{tercile_num,1} = append_rows(data_value, tercile_means{tercile_num, 1});
        end
        
        %  do not care about central trials, use the
        % center spot for the purposes of figuring out the mean of both
        % peripheral trials
        mean_deviation = table_column_mean(append_rows(trials_by_tercile{1, 1}, trials_by_tercile{3, 1}), 2);
        mean_centering = table_column_mean(append_rows(trials_by_tercile{1, 1}, trials_by_tercile{3, 1}), 1);
        data_value = {subject_name mean_centering mean_deviation};
        tercile_means{2,1} = append_rows(data_value, tercile_means{2, 1});
    end
    datatables{group_idx, 1} = tercile_means;
end

% Datatable 3 and 4 - number of trials for each subject.
% For each group, we will have a table in the following form:
% Subject | Number of Trials
for group_idx = (1:2)
    group = data{group_idx, 1};
    datatable_number = group_idx + 2;
    datatables{datatable_number, 1} = {'Subject Name', 'Number of Trials'};
    for subject_idx = (1:size(group.subjects, 1))
        subject_name = group.subjects{subject_idx, 1}.name;
        subject_trials = size(group.subjects{subject_idx, 1}.trials, 1);
        datatables{datatable_number, 1} = append_rows(datatables{datatable_number, 1}, {subject_name, subject_trials});
    end
end

% End of setup of the data.

% Figure One (2) - Breaking the arrow plot into two subplots. Otherwise, it is
% the same thing as we have done in the past.
figure;
extrema = zeros(2);

plotting_arrows = cell(2,1);
start_points_list = [0 0];

average_tercile_size = cell(2, 1);

for group_idx = (1:2)
    subplot(1, 2, group_idx);
    hold on;
    group = data{group_idx, 1};
    group_name = group.name{1,1};
    title(group_name, 'fontweight', 'bold', 'fontsize', 16);
    xlabel('frequency in cents', 'fontweight', 'bold', 'fontsize', 16);
    ylabel('subject', 'fontweight', 'bold', 'fontsize', 16);
    
    average_tercile_size_group = zeros(size(group.subjects, 1), 1);
    
    % for each subject within this group
    for subject_idx = (1:size(group.subjects, 1))
        subject = group.subjects{subject_idx, 1};
        % for the purposes of this figure, do not care about subject
        % names. Only the subject index;
        trials = subject.trials;
        
        trials_by_tercile = cell(3, 1);
        
        % figure out the "effective deviation" and store in terciles.
        for trial_idx = (1:size(trials, 1))
            % calculate the endpoint of each trial
            trial = subject.trials(trial_idx, :);
            polarity = trial(1, 3);
            polarity = polarity{1,1};
            polarity = polarity{1,1};
            if(isequal(polarity, 'NEGATIVE'))
                deviation = trial{1, 2}(1,1) * -1;
                centering = trial{1,1}(1,1);
            else
                deviation = trial{1,2}(1,1);
                centering = trial{1, 1}(1,1) * -1;
            end
            
            endpoint = deviation + centering;
            trial{1,6} = deviation;
            trial{1, 7} = endpoint;
            
            tercile_num = trial{1,5}(1,1);
            trials_by_tercile{tercile_num, 1} = append_rows(trial, trials_by_tercile{tercile_num, 1});
        end
        
        % calculate the overall mean values.
        mean_motion = zeros(3, 2);
        for tercile_num = (1:3)
            mean_motion(tercile_num, 1) = mean(cellfun(@(c) c(1,1), trials_by_tercile{tercile_num, 1}(:, 6)));
            
            % append this value to our list of deviations, use it for the
            % calculation of the outliers.
            start_points_list(size(start_points_list(:, group_idx), 1) + 1, group_idx) = mean_motion(tercile_num, 1);

            mean_motion(tercile_num, 2) = mean(cellfun(@(c) c(1,1), trials_by_tercile{tercile_num, 1}(:, 7)));
        end
        
        % mean motion shows the start point to end point difference.
        % Analyzing mean motion can give information about the terciles
        tercile_sizes = abs(mean_motion(:, 2) - mean_motion(:, 1));
        average_tercile_size_group(subject_idx, 1) = mean(tercile_sizes);
        
        
        
        plotting_arrows{group_idx, 1}{subject_idx, 1} = mean_motion;
        
        if(extrema(group_idx,  1)< max(max(mean_motion)))
            extrema(group_idx, 1) = max(max(mean_motion));
        end
        if(extrema(group_idx, 2) > min(min(mean_motion)))
            extrema(group_idx, 2) = min(min(mean_motion));
        end
        
        
    end
    average_tercile_size{group_idx, 1} = average_tercile_size_group;
    
    hold off;
    
    set(gca,'YTick',[]);
    
    
end
% delete the first row, as it had placeholder zeros anyway 

start_points_list(1, :) = nan;

mean_start_points(1) = 0; %nanmean(start_points_list(:, 1));
mean_start_points(2) = 0; %nanmean(start_points_list(:, 2));

std_start_points(1) = nanstd(abs(start_points_list(:, 1)));
std_start_points(2) = nanstd(abs(start_points_list(:, 2)));

% see which group has the largest range... this is determined by the
% standard deviation
   % group one has the larger range 
   rect_startx(1) = mean_start_points(1) - std_start_points(1);
   rectw(1) = 2 * std_start_points(1);
    % group two has the larger range
   rect_startx(2) = mean_start_points(2) - std_start_points(2);
   rectw(2) = 2 * std_start_points(2);

% on BOTH of the diagrams, draw the rectangle 
for i = (1:2)
    subplot(1,2,i);
    ymin = 0;
    ymax = size(data{2, 1}.subjects, 1) + 1;
    % rectangle was unhelpful; disabling plotting temporarily. 
    % rectangle('Position', [rect_startx(i) ymin rectw(i) ymax], 'FaceColor', [0.9 0.9 0.9]);
end


% Draw the arrows
for group_idx = (1:2)
    subplot(1,2,group_idx);
    
    % get our order based on the average tercile size (greatest to
    % smallest)
    
    average_tercile_size_by_group(:, 1) = average_tercile_size{group_idx, 1};
    average_tercile_size_by_group(:, 2) = 1:size(data{group_idx, 1}.subjects);
    
    average_tercile_size_by_group = sortrows(average_tercile_size_by_group, 1, 'descend');
   
   
    
    hold on;
    for subject_idx = (1:size(data{group_idx, 1}.subjects, 1))
        mean_motion = plotting_arrows{group_idx, 1}{average_tercile_size_by_group(subject_idx, 2), 1};
        quick_arrow(mean_motion(2, 1), subject_idx, mean_motion(2,2), subject_idx, [0.5 0.5 0.5]);
        quick_arrow(mean_motion(1, 1), subject_idx, mean_motion(1,2), subject_idx, [1 0 0]);
        quick_arrow(mean_motion(3, 1), subject_idx, mean_motion(3,2), subject_idx, [0 0 1]);
    end
    
    hold off;
end

hold off;

% Figure Two : centering bars with error bar.
% group_1 lower | group_2 lower | group1 upper | group 2 upper
figure;

group_one_name = data{1, 1}.name{1,1};
group_two_name = data{2, 1}.name{1,1};

plotting_data = zeros(2, 4);
plot_idx = 1;
table_idx = 1;
while(plot_idx <= 2)
    plotting_data(plot_idx, 1) = table_column_mean(datatables{1,1}{table_idx, 1}, 2);
    plotting_data(plot_idx, 2) = table_column_mean(datatables{2,1}{table_idx, 1}, 2);
    
    plotting_data(plot_idx, 3) = table_column_error(datatables{1,1}{table_idx, 1}, 2);
    plotting_data(plot_idx, 4) = table_column_error(datatables{2,1}{table_idx, 1}, 2);
    
    table_idx = table_idx + 2;
    plot_idx = plot_idx + 1;
end

hold on;

ylabel('centering (cents)', 'fontweight', 'bold', 'fontsize', 24);

set(gca,'XTick',[1, 2, 3, 4]);
set(gca, 'XTickLabel', {strcat(group_one_name, ' Lower'), strcat(group_two_name, ' Lower'), strcat(group_one_name, ' Upper'), strcat(group_two_name, ' Upper')}, 'fontweight', 'bold', 'fontsize', 16);

% graphicz
% bars
% bars
bar(1, plotting_data(1,1), 'c');
bar(2, plotting_data(1,2), 'g');
bar(3, plotting_data(2,1), 'c');
bar(4, plotting_data(2,2), 'g');

% error bars

errorbar(1, plotting_data(1,1), plotting_data(1,3), plotting_data(1,3));
errorbar(2, plotting_data(1,2), plotting_data(1,4), plotting_data(1,4));
errorbar(3, plotting_data(2,1), plotting_data(2,3), plotting_data(2,3));
errorbar(4, plotting_data(2,2), plotting_data(2,4), plotting_data(2,4));
hold off;

% Figure Three : deviation bars with error bar.
figure;

group_one_name = data{1, 1}.name{1,1};
group_two_name = data{2, 1}.name{1,1};

plotting_data = zeros(2, 4);
plot_idx = 1;
table_idx = 1;
while(plot_idx <= 2)
    plotting_data(plot_idx, 1) = table_column_mean(datatables{1,1}{table_idx, 1}, 3);
    plotting_data(plot_idx, 2) = table_column_mean(datatables{2,1}{table_idx, 1}, 3);
    
    plotting_data(plot_idx, 3) = table_column_error(datatables{1,1}{table_idx, 1}, 3);
    plotting_data(plot_idx, 4) = table_column_error(datatables{2,1}{table_idx, 1}, 3);
    
    table_idx = table_idx + 2;
    plot_idx = plot_idx + 1;
end

hold on;

ylabel('deviation (cents)', 'fontweight', 'bold', 'fontsize', 24);

set(gca,'XTick',[1, 2, 3, 4]);
set(gca, 'XTickLabel', {strcat(group_one_name, ' Lower'), strcat(group_two_name, ' Lower'), strcat(group_one_name, ' Upper'), strcat(group_two_name, ' Upper')}, 'fontweight', 'bold', 'fontsize', 16);

% graphicz
% bars
bar(1, plotting_data(1,1), 'c');
bar(2, plotting_data(1,2), 'g');
bar(3, plotting_data(2,1), 'c');
bar(4, plotting_data(2,2), 'g');

% error bars

errorbar(1, plotting_data(1,1), plotting_data(1,3), plotting_data(1,3));
errorbar(2, plotting_data(1,2), plotting_data(1,4), plotting_data(1,4));
errorbar(3, plotting_data(2,1), plotting_data(2,3), plotting_data(2,3));
errorbar(4, plotting_data(2,2), plotting_data(2,4), plotting_data(2,4));
hold off;


% Figure Four : centering/deviation bars with error bar.
figure;

group_one_name = data{1, 1}.name{1,1};
group_two_name = data{2, 1}.name{1,1};

plotting_data = zeros(2, 4);
plot_idx = 1;
table_idx = 1;
while(plot_idx <= 2)
    plotting_data(plot_idx, 1) = nanmean(table_to_vector(datatables{1,1}{table_idx, 1}, 2)./table_to_vector(datatables{1,1}{table_idx, 1}, 3));
    plotting_data(plot_idx, 2) = nanmean(table_to_vector(datatables{2,1}{table_idx, 1}, 2)./table_to_vector(datatables{2,1}{table_idx, 1}, 3));
    
    plotting_data(plot_idx, 3) = nanstd(table_to_vector(datatables{1,1}{table_idx, 1}, 2)./table_to_vector(datatables{1,1}{table_idx, 1}, 3))/sqrt(size(table_to_vector(datatables{1,1}{table_idx, 1}, 2)./table_to_vector(datatables{1,1}{table_idx, 1}, 3), 1));
    plotting_data(plot_idx, 4) = nanstd(table_to_vector(datatables{2,1}{table_idx, 1}, 2)./table_to_vector(datatables{2,1}{table_idx, 1}, 3))/sqrt(size(table_to_vector(datatables{2,1}{table_idx, 1}, 2)./table_to_vector(datatables{2,1}{table_idx, 1}, 3), 1));
    
    table_idx = table_idx + 2;
    plot_idx = plot_idx + 1;
end

hold on;

ylabel('centering/deviation ratio', 'fontweight', 'bold', 'fontsize', 24);

set(gca,'XTick',[1, 2, 3, 4]);
set(gca, 'XTickLabel', {strcat(group_one_name, ' Lower'), strcat(group_two_name, ' Lower'), strcat(group_one_name, ' Upper'), strcat(group_two_name, ' Upper')}, 'fontweight', 'bold', 'fontsize', 16);

% graphicz
% bars
% bars
bar(1, plotting_data(1,1), 'c');
bar(2, plotting_data(1,2), 'g');
bar(3, plotting_data(2,1), 'c');
bar(4, plotting_data(2,2), 'g');

% error bars

errorbar(1, plotting_data(1,1), plotting_data(1,3), plotting_data(1,3));
errorbar(2, plotting_data(1,2), plotting_data(1,4), plotting_data(1,4));
errorbar(3, plotting_data(2,1), plotting_data(2,3), plotting_data(2,3));
errorbar(4, plotting_data(2,2), plotting_data(2,4), plotting_data(2,4));
hold off;


debug = data;
output = datatables;

% Figure Five : Deviation Histogram.
current_group = datatable{2,6};
colidx = 1;
rowpos = 1;
group_name = {};
% overallocate signed deviations
signed_deviations = nan(size(datatable, 1) - 1, 2);

for row_idx = (2:size(datatable, 1))
    deviation = datatable{row_idx, 3};
    if(strcmp(datatable{row_idx, 4}, 'NEGATIVE'))
       % flip our deviation
       deviation = -1*deviation;
    end
    if(~strcmp(datatable{row_idx, 6},current_group))
        group_name{colidx} = current_group;
        colidx = colidx + 1;
        current_group = datatable{row_idx, 6};
        rowpos = 1;
    end
   signed_deviations(rowpos, colidx) = deviation;
   rowpos = rowpos + 1;
end
group_name{colidx} = current_group;

figure;
subplot(2, 1, 1);

binsize = 30;

range = max(signed_deviations(:, 1)) - min(signed_deviations(:, 1));
hist(signed_deviations(:, 1), range/binsize);
xlim([-500 500]);
xlabel('deviations (cents)', 'fontweight', 'bold', 'fontsize', 16);
title(group_name{1}, 'fontweight', 'bold', 'fontsize', 16);

subplot(2, 1, 2);
range = max(signed_deviations(:, 2)) - min(signed_deviations(:, 2));
hist(signed_deviations(:, 2), range/binsize);
xlim([-500 500]);
xlabel('deviations (cents)', 'fontweight', 'bold', 'fontsize', 16);
title(group_name{2}, 'fontweight', 'bold', 'fontsize', 16);


end
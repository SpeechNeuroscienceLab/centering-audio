% Anantajit Subrahmanya
% September 20 2020
% Simple utility to find the mean for a column
% PARAM table - cell array with numeric values in a  column
% PARAM col - index of the column that we want to find mean of
% RETURN error - the mean of the 'col'th column of the table

function [average] = table_column_mean(table, col)
    total_value = 0;
    nan_count = 0;
    for rownum = (1:size(table, 1))
        if(isnan(table{rownum, col}))
            nan_count = nan_count + 1;
            continue;
        end
        total_value = total_value + table{rownum, col};
    end
    average = total_value/(size(table, 1) - nan_count);
end
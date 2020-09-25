% Anantajit Subrahmanya
% September 20 2020
% Simple utility to find the standard error for a column
% PARAM table - cell array with numeric values in a  column
% PARAM col - index of the column that we want to find standard err of
% RETURN error - the standard error of the 'col'th column of the table

function [error] = table_column_error(table, col)
    cleaned_up_data = zeros(size(table, 1), 1);
    for rownum = (1:size(table, 1))
        cleaned_up_data(rownum, 1) = table{rownum, col};
    end
    error = nanstd(cleaned_up_data)/sqrt(size(cleaned_up_data, 1));
end
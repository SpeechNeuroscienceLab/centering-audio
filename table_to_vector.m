% Anantajit Subrahmanya
% September 20 2020
% Simple utility to convert a datatable into a vector of doubles
% PARAM table - cell array with numeric values in a  column
% RETURN vector - matrix vwersion of a cell table of numerical values.  
function [vector] = table_to_vector(table, col)
    vector = zeros(size(table, 1), 1);
    for rownum = (1:size(table, 1))
        vector(rownum, 1) = table{rownum, col};
    end
end
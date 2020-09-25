% Anantajit Subrahmanya
% September 20 2020
% Simple utility to append rows to a table 
% PARAM original - a k by n cell array 
% PARAM addition - a l by n cell array
% RETURN table that concats into a (k+l) by n cell array

function [table] = append_rows(original, addition)
    table = original;
    
    number_of_rows = size(original, 1);
    
    number_of_original_cols = size(original, 2);
    
    number_of_cols_to_add = size(addition, 2);
    
    for row_num = (1:number_of_rows)
        for col_num = (1: number_of_cols_to_add)
            table{row_num, col_num + number_of_original_cols} = addition(row_num, col_num);
        end
    end
    return;
end
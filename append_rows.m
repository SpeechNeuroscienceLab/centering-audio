% Anantajit Subrahmanya
% September 20 2020
% Simple utility to append rows to a table 
% PARAM original - a k by n cell array 
% PARAM addition - a k by m cell array
% RETURN table that concats into a k by (n + m) cell array
function [table] = append_rows(original, addition)
    table = original;
    
    number_of_columns = size(original, 2);
    
    number_of_original_rows = size(original, 1);
    
    number_of_rows_to_add = size(addition, 1);
    
    row_num = 1;
    while(row_num <= number_of_rows_to_add)
       col_num = 1;
       
       while(col_num <= number_of_columns)
          table{number_of_original_rows + row_num, col_num} = addition{row_num, col_num};
          col_num = col_num + 1;
       end
       row_num = row_num + 1;
    end
    
    return;
end
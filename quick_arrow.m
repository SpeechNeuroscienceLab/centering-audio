% Anantajit Subrahmanya
% September 20 2020
% Simple utility to draw arrows, with tail at the intitial position and the
%   tail at the final position on an active subplot
% PARAM - x1 (initial x position)
% PARAM - y1 (initial y position)
% PARAM - x2 (final x position)
% PARAM - y2 (final y position)
% COLOR - double array that contains [R,G,B] value of arrow

function quick_arrow(x1, y1, x2, y2, color)

% possible arrow styles are: arrow, flag
arrow_style = 'flag';

% not used anymore... at least for pitch
if(strcmp(arrow_style, 'arrow'))
    hold on;
    plot([x1 x2], [y1 y2], 'color', color);
    
    arrowhead_size = 1;
    
    
    if(x2 - x1 < 0)
        direction = -1;
    else
        direction = 1;
    end
    
    arrowhead_points{1} = [x2, y2 + arrowhead_size/5];
    arrowhead_points{2} = [x2 + 5*arrowhead_size*direction, y2];
    arrowhead_points{3} = [x2, y2 - arrowhead_size/5];
    
    previous_point = [x2 y2];
    for i = (1:3)
        current_point = arrowhead_points{i};
        plot([previous_point(1) current_point(1)], [previous_point(2) current_point(2)], 'color', color);
        previous_point = current_point;
    end
    current_point = [x2 y2];
    plot([previous_point(1) current_point(1)], [previous_point(2) current_point(2)], 'color', color);
    
    scatter(x1, y1, [], color);
    hold off;
end




if(strcmp(arrow_style, 'flag'))
    hold on;
    
    arrowhead_size = 1/5;
    line_width = 2;
    
    
    plot([x2 x1], [y1 y2 - arrowhead_size], 'color', color, 'LineWidth', line_width);
    plot([x1 x1], [y2 - arrowhead_size y2 + arrowhead_size], 'color', color, 'LineWidth', line_width);
    plot([x1 x2], [y2 + arrowhead_size y1], 'color', color, 'LineWidth', line_width);
    
    hold off;
end
end
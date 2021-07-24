function [arg1out,arg2out] = my_figure(varargin)
% function [hf,[yes_dual_monitor]] = my_figure(...)
% takes all the same args as figure,
% but optionally lowers figure (to deal with 2nd monitor issue)

global yes_dual_monitor

if isempty(yes_dual_monitor)
  reply = input('dual monitor? y/[n]: ','s');
  if strcmp(reply,'y')
    yes_dual_monitor = 1;
  else
    yes_dual_monitor = 0;
  end
end
if nargout >= 1
  arg1out = figure(varargin{:});
else
  figure(varargin{:});
end
if yes_dual_monitor
  lower_figure;
end

if nargout >= 2
  arg2out = yes_dual_monitor;
end

  
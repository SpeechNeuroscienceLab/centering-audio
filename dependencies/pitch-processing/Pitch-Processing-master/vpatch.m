function hp = vpatch(x1,x2,colorspec)
% function hp = vpatch(x1,x2,colorspec)
% To make gray patch:
% if ~isstr(colorspec) && (length(colorspec) == 1)
%   color4patch = colorspec*[1 1 1];
% else
%   color4patch = colorspec;

if ~isstr(colorspec) && (length(colorspec) == 1)
  color4patch = colorspec*[1 1 1];
else
  color4patch = colorspec;
end
a = axis;
a(4) = 0.95*a(4);
hp = patch([x1 x2 x2 x1 x1],[a(3) a(3) a(4) a(4) a(3)],color4patch);
set(hp,'LineStyle','none');

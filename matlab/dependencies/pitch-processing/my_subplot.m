function hsubplot = my_subplot(nrows,ncols,isubplot,fig_bord,subplot_topbord,subplot_rightbord,isubplot_rot,isubplot_fliprow,isubplot_flipcol)
% function hsubplot = my_subplot(nrows,ncols,isubplot,[fig_bord],[subplot_topbord],[subplot_rightbord],[isubplot_rot],[isubplot_fliprow],[isubplot_flipcol])

if nargin < 4 || isempty(fig_bord), fig_bord = 0.1; end
if nargin < 5 || isempty(subplot_topbord), subplot_topbord = 0.05; end
if nargin < 6 || isempty(subplot_rightbord), subplot_rightbord = 0.05; end
if nargin < 7 || isempty(isubplot_rot), isubplot_rot = 0; end
if nargin < 8 || isempty(isubplot_fliprow), isubplot_fliprow = 0; end
if nargin < 9 || isempty(isubplot_flipcol), isubplot_flipcol = 0; end

if isubplot > nrows*ncols
  error(sprintf('my_subplot(%d,%d,[]) can only accomodate %d plots, not %d plots', ...
		nrows,ncols,nrows*ncols,isubplot));
end

icol = my_mod(isubplot,ncols);
irow = ceil(isubplot/ncols);

if isubplot_fliprow, irow = nrows - irow + 1; end
if isubplot_flipcol, icol = ncols - icol + 1; end

actual_isubplot_rot = round((360/(2*pi))*angle(exp(i*isubplot_rot*(2*pi)/360)));
switch actual_isubplot_rot
  case 0
    actual_nrows = nrows;
    actual_ncols = ncols;
    actual_irow = irow;
    actual_icol = icol;
  case 90
    actual_nrows = ncols;
    actual_ncols = nrows;
    actual_irow = ncols - icol + 1;
    actual_icol = irow;
  case -90
    actual_nrows = ncols;
    actual_ncols = nrows;
    actual_irow = icol;
    actual_icol = nrows - irow + 1;
  case 180
    actual_nrows = nrows;
    actual_ncols = ncols;
    actual_irow = nrows - irow + 1;
    actual_icol = ncols - icol + 1;
  case -180
    actual_nrows = nrows;
    actual_ncols = ncols;
    actual_irow = nrows - irow + 1;
    actual_icol = ncols - icol + 1;
  otherwise
    error('isubplot_rot(%d) not allowed, must be 180, 90, 0[the default], -90, or -180', isubplot_rot);
end
actual_isubplot = (actual_irow-1)*actual_ncols + actual_icol;
% fprintf('%d:(%d,%d) actual: (%d,%d):%d\n',isubplot,irow,icol,actual_irow,actual_icol,actual_isubplot); % for debug

fig_xorg = fig_bord;
fig_yorg = fig_bord;
fig_width  = 1.0 - 2*fig_bord;
fig_height = 1.0 - 2*fig_bord;

subplot_outerwidth = fig_width/actual_ncols;
subplot_outerheight = fig_height/actual_nrows;

subplot_width  = subplot_outerwidth  - subplot_rightbord/actual_ncols;
subplot_height = subplot_outerheight - subplot_topbord/actual_nrows;

subplot_row = ceil(actual_isubplot/actual_ncols);
subplot_col = actual_isubplot - (subplot_row - 1)*actual_ncols;

subplot_ncols_toleft = subplot_col - 1;
subplot_nrows_below  = actual_nrows - subplot_row;

subplot_xorg = fig_xorg + subplot_ncols_toleft * subplot_outerwidth;
subplot_yorg = fig_yorg + subplot_nrows_below * subplot_outerheight;

subplot_pos = [subplot_xorg subplot_yorg subplot_width subplot_height];

hsubplot = axes;
set(hsubplot,'Position',subplot_pos);
set(hsubplot,'ActivePositionProperty','Position');
% set(hsubplot,'Visible','off');

function move2back(hax_or_h,h)
% function moveback([hax],h)
% move h to the back of hax

if nargin < 2
    h = hax_or_h;
    hax = gca;
elseif isempty(hax_or_h)
    hax = gca;
else
    hax = hax_or_h;
end

axkids = get(hax,'Children');
ih = dsearchn(axkids,h);
if isempty(ih), error('handle(%f) not a child of hax(%f)', h, hax); end
remh_axkids = axkids;
remh_axkids(ih) = [];
new_axkids = [remh_axkids; h];
set(hax,'Children',new_axkids);

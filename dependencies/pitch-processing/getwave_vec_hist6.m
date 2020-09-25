function tmp_cat = getwave_vec_hist6(vec_hist,itrial)
% function tmp_cat = getwave_vec_hist6(vec_hist,itrial)
% get the data from trial(itrial) of a new_quatier6-type vec_hist
% 'tmp_cat' is the full time waveform, peiced together from the data frames

if ~vec_hist.playable
  reply = input(sprintf('vec_hist(%s) is not playable, override? y/[n]: ',vec_hist.name),'s')
  if isempty(reply) || reply == 'n'
    error(sprintf('vec_hist(%s) is not playable',vec_hist.name));
  end
end

trial_vec_hist = squeeze(vec_hist.data(itrial,:,:));

tmp = trial_vec_hist';
tmp_cat = tmp(:);

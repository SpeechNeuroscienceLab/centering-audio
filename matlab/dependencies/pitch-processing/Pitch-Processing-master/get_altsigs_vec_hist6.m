function altsigs = get_altsigs_vec_hist6(vec_hist,itrial,data_size)
% function altsigs = get_altsigs_vec_hist6(vec_hist,itrial,,data_size)

if nargin < 3
  altsigs = squeeze(vec_hist.data(itrial,:,:));
else
  altsigs_frames = squeeze(vec_hist.data(itrial,:,:));
  nframes = vec_hist.nvecs;
  nsigs = vec_hist.vec_size;
  nsamps = nframes*data_size;
  altsigs = zeros(nsamps,nsigs);
  for ifr = 1:nframes
    isamp_llim = (ifr-1)*data_size + 1;
    isamp_ulim = ifr*data_size;
    altsigs(isamp_llim:isamp_ulim,:) = repmat(altsigs_frames(ifr,:),data_size,1);
  end
end


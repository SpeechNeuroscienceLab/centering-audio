function pitch = pitch_v2_vec_hist6(vec_hist,itrial,pitchlimits,fs,yanalframelen_suggest)
% function pitch = pitch_vec_hist6(vec_hist,itrial,[pitchlimits],[fs],[yanalframelen])
% e.g., pitchlimits = [70 300], and
% for fs = 11025, yanalframelen = 400 are good values (and are the defaults)

if ~vec_hist.playable
  error(sprintf('vec_hist(%s) is not playable',vec_hist.name));
end

if nargin < 5 || isempty(yanalframelen_suggest)
  yanalframelen_suggest = 400;
end
if nargin < 4 || isempty(fs)
  fs = 11025;
end
if nargin < 3 || isempty(pitchlimits)
  pitchlimits = [70 300]; % Hz limits for pitch
end

ystep = vec_hist.vec_size;
nframes = vec_hist.nvecs;
trial_yframes = squeeze(vec_hist.data(itrial,:,:));

ishortest_period = round(fs/pitchlimits(2));
ilongest_period = round(fs/pitchlimits(1));

yanalframelen = yanalframelen_suggest - rem(yanalframelen_suggest,2);
Yanalframe = zeros(yanalframelen,1);
pitch = zeros(1,nframes);
for ifr = 1:nframes
  Ynewchunk = trial_yframes(ifr,:)';
  Yanalframe = [Yanalframe((ystep+1):yanalframelen); Ynewchunk];
  fullyac = xcorr(Yanalframe);
  yac = fullyac(yanalframelen:end);
  yacsel = yac(ishortest_period:ilongest_period);
  [ipyac,npyac] = peakfind(yacsel);
  if npyac > 0
    apyac = yacsel(ipyac);
    [max_apyac,imax_apyac] = max(apyac);
    ipyac2use = ipyac(imax_apyac);
    [pyac2use,duh] = fitpeak(yacsel,ipyac2use);
    % the final '- 1' below is because yac(1) represents lag 0 in the autocorrelation
    pitch_period = (pyac2use + ishortest_period - 1) - 1; 
    pitch(ifr) = fs/pitch_period;
    % in magspec, ipitch = pitch/freqsep (ipitch is a fractional number)
    % then pitch harmonics at n*ipitch + 1, for n = 1,2,...
    % use this to compute harmonics-to-noise ratio from magspec
  else
    apyac = [];
    max_apyac = 0;
    imax_apyac = 0;
    ipyac2use = 0;
    pyac2use = 0;
    pitch_period = 0;
    pitch(ifr) = 0;
  end
end

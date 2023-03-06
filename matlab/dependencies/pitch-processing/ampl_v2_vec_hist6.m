function ampl = ampl_v2_vec_hist6(vec_hist,itrial,fs,ms_akern,expon)
% function ampl = ampl_v2_vec_hist6(vec_hist,itrial,fs,ms_akern,expon)

if nargin < 3 || isempty(fs), fs = 11025; end
if nargin < 4 || isempty(ms_akern), ms_akern = 30; end
if nargin < 5 || isempty(expon), expon = 2; end

trial_vec_hist = squeeze(vec_hist.data(itrial,:,:));
tmp = trial_vec_hist';
sig = tmp(:);
amplsig = get_sig_ampl(sig,fs,ms_akern,expon,0);

ystep = vec_hist.vec_size;
nframes = vec_hist.nvecs;
ampl = zeros(1,nframes);
for ifr = 1:nframes
  ampl(ifr) = mean(amplsig(((ifr-1)*ystep+1):(ifr*ystep)));
end

function audio_buffer = get_audio_buffer_vec_hist6(audio_buffer_vechist_name,iblock,n_audio_chans)

set_vec_types;

if nargin < 2, iblock = []; end
if nargin < 3 || isempty(n_audio_chans), n_audio_chans = 4; end

abuf_vechist = get_vec_hist6(audio_buffer_vechist_name,SHORT_VEC,iblock);

ntrials = abuf_vechist.ntrials;
nframes = abuf_vechist.nvecs;
data_frame_size = round((abuf_vechist.vec_size)/n_audio_chans);
if data_frame_size*n_audio_chans ~= abuf_vechist.vec_size, error('wrong number of audio channels specified'); end
nsamps = nframes*data_frame_size;

a = reshape(abuf_vechist.data,[ntrials nframes n_audio_chans data_frame_size]);
audio_buffer = zeros(ntrials,n_audio_chans,nsamps);
for itrial = 1:ntrials
  for ichan = 1:n_audio_chans
    tmp = squeeze(a(itrial,:,ichan,:))';
    audio_buffer(itrial,ichan,:) = tmp(:);
  end
end

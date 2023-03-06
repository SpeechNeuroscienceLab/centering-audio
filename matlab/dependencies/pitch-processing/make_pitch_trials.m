function make_pitch_trials(vods,pitchlimits,yes_plot_trials)


if nargin < 2 || isempty(yes_plot_trials), yes_plot_trials = 1; end

fs = 11025;
nsamps_frame = 32;
ffr = fs/nsamps_frame;
[B,A] = butter(5,(20/(ffr/2)));
plot_cents_range = [-250 250];

clear vec_hist
vec_hist(1) = get_vec_hist6('inbuffer',3);
vec_hist(2) = get_vec_hist6('outbuffer',3);
vec_hist(3) = get_vec_hist6('blockalt',3);
% Added by Hardik 12/05/2014
voice_onset=strcmpi(vods,'y');
if voice_onset==1
    
% added by Gill 01/16/2013  % % % % % % % % % % % % % % % % % % % % % %
abo = get_audio_buffer_vec_hist6('audio_buffer_out');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% added by jfh 02/12/14
vec_hist(4) = get_vec_hist6('voice_onset_detect',2);
vec_hist(5) = get_vec_hist6('blockalt',3);
voice_onset_detect_sig = vec_hist(4).data;
pert_alt_sig = squeeze(vec_hist(5).data(:,:,1));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
end
%%%%%%%%%%%%%%%%%
ntrials = find_ntrials_in_vechist(vec_hist(1));
nframes = vec_hist(1).nvecs;
ystep = vec_hist(1).vec_size;
taxis_frame = (0:(nframes-1))*ystep/fs;
taxis = (0:((nframes*ystep)-1))/fs;

raw_pitch_in = zeros(ntrials,nframes);
raw_pitch_out = zeros(ntrials,nframes);
pitch_in = zeros(ntrials,nframes);
pitch_out = zeros(ntrials,nframes);
ampl_in = zeros(ntrials,nframes);
ampl_out = zeros(ntrials,nframes);
pitch_cents_in = zeros(ntrials,nframes);
pitch_cents_out = zeros(ntrials,nframes);
pitch_pert = zeros(ntrials,nframes);

if yes_plot_trials, my_figure; end
% Added by Hardik 12/05/2014
if voice_onset==1
thisblockbadtrials = zeros(1,ntrials);
end
%%%%%%%%%%%%%%%%%%%%
for itr = 1:ntrials
    wave_in(itr,:) = getwave_vec_hist6(vec_hist(1),itr);
    raw_pitch_in(itr,:)  = pitch_v2_vec_hist6(vec_hist(1),itr,pitchlimits,fs,400);
    ampl_in(itr,:)  =  ampl_v2_vec_hist6(vec_hist(1),itr,fs);
    pitch_in(itr,:)  = filtfilt(B,A,raw_pitch_in(itr,:));
    wave_out(itr,:) = getwave_vec_hist6(vec_hist(2),itr);
    raw_pitch_out(itr,:) = pitch_v2_vec_hist6(vec_hist(2),itr,pitchlimits,fs,400);
    ampl_out(itr,:) =  ampl_v2_vec_hist6(vec_hist(2),itr,fs);
    pitch_out(itr,:) = filtfilt(B,A,raw_pitch_out(itr,:));
    altsigs = get_altsigs_vec_hist6(vec_hist(3),itr);
%     if itr == 10
%         warning('poop');
%     end
% Added by Hardik 12/05/2014
if voice_onset==1
    % added by Gill on 1/16/2013  % % % % % % % % % % % % % % % %
    thistrialaltsigs = altsigs(:,2)';

%    thistrialabo = squeeze(abo(itr,3,:))';
%    
%    time_onset_pert = find(diff(thistrialabo),1) + 1;
%    time_offset_pert = find(diff(thistrialabo),1,'last');
%    
%    frame_onset_pert = round(time_onset_pert/nsamps_frame);
%    frame_offset_pert = round(time_offset_pert/nsamps_frame);

    % added by jfh 02/12/14

    frame_onset_pert = find(voice_onset_detect_sig(itr,:),1) + find(pert_alt_sig(itr,:),1);
    frame_offset_pert = find(voice_onset_detect_sig(itr,:),1,'last') + find(pert_alt_sig(itr,:),1,'last');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    subframelength = frame_offset_pert - frame_onset_pert;
    
    onsetblockpert = find(diff(thistrialaltsigs),1) + 1;
    offsetblockpert = find(diff(thistrialaltsigs),1,'last');
    subblockpert = offsetblockpert - onsetblockpert;
    
    absaltsigsthistrial = abs(thistrialaltsigs);
    [maxabsaltsigs, maxabsaltsigsindex]  = max(absaltsigsthistrial);
    pertdirectionval = thistrialaltsigs(1, maxabsaltsigsindex);
    
    vodaltsigs = zeros(1,length(thistrialaltsigs));
    if subframelength ~= subblockpert
        difflengths2 = subblockpert - subframelength;
        check_frame_offset_pert = frame_offset_pert + difflengths2;
        
        if check_frame_offset_pert > length(thistrialaltsigs)
            frame_offset_pert = length(thistrialaltsigs);
            %warning('Perturbation for this trial extends beyond the end of the trial. Subject most likely spoke too late.');
            thisblockbadtrials(1,itr) = 1;
        else
            frame_offset_pert = frame_offset_pert + difflengths2;
        end
    end
    
    vodaltsigs(1,frame_onset_pert:frame_offset_pert) = pertdirectionval;
    vodaltsigs = vodaltsigs';
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
end
%%%%%%%%%%%%%%%%%%
% Added by Hardik 12/05/2014
if voice_onset==0
 pitch_pert(itr,:) = altsigs(:,2);
end
if voice_onset==1
    pitch_pert(itr,:) = vodaltsigs;
end
%%%%%%%%%%%%%%%%%%
    if yes_plot_trials
        clf
        subplot(3,1,1)
        plot(taxis,wave_in(itr,:))
        a = axis; axis([taxis(1) taxis(end) a(3:4)]);
        title(sprintf('trial(%d)',itr));
        subplot(3,1,2)
        plot(taxis_frame,raw_pitch_out(itr,:),'b')
        hold on
        plot(taxis_frame,raw_pitch_in(itr,:),'r')
        hold off
        a = axis; axis([taxis_frame(1) taxis_frame(end) a(3:4)]);
        subplot(3,1,3)
        plot(taxis_frame,ampl_out(itr,:),'b')
        hold on
        plot(taxis_frame,ampl_in(itr,:),'r')
        hold off
        a = axis; axis([taxis_frame(1) taxis_frame(end) a(3:4)]);
        pause
    end
    fprintf('.');
end
fprintf('\n');

save('pitch_trials','fs','pitchlimits','ntrials','nframes','ystep','taxis','taxis_frame', ...
    'wave_in','wave_out','pitch_in','pitch_out','ampl_in','ampl_out','pitch_pert');

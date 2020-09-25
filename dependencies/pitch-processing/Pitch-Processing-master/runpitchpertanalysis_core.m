subjdir = subj_info(isubj).subjdir;
pitchlimits = subj_info(isubj).pitchlimits;
consolidate_audiodir = subj_info(isubj).speak_consolidate_audiodir;
speak_audiodirs = subj_info(isubj).speak_audiodirs;
  
nblocks = length(speak_audiodirs);

curdir = cd;
%cd(subjdir)
curdir_subj = cd;
parse_fig_pos = [1000  371  991 1127];
if ~exist('reject_pitchlimits','var'), reject_pitchlimits = [50 300]; end


yes_add_ampl_info = 0;
tic
parfor iblock = 1:nblocks
  expr_audiodir = speak_audiodirs{iblock};
  cd(expr_audiodir); fprintf('cd expr_audiodir(%s)\n',expr_audiodir);
  if yes_add_ampl_info || (yes_overwrite >= 6) || ~exist('pitch_trials.mat','file'), make_pitch_trials(vods,pitchlimits,0); end
  cd(curdir_subj);
end
toc
cd(consolidate_audiodir); fprintf('cd consolidate_audiodir(%s)\n',consolidate_audiodir);
if yes_add_ampl_info || (yes_overwrite >= 5) || ~exist('pitch_trials.mat','file')
  cd(curdir_subj)
  make_consolidated_pitch_trials(consolidate_audiodir,speak_audiodirs,pitchlimits);
  cd(consolidate_audiodir); fprintf('cd consolidate_audiodir(%s)\n',consolidate_audiodir);
end
load('pitch_trials');
% reject_pitchlimits = [200 300] % for expr20140213...run1mac
if (yes_overwrite >= 4) || ~exist('parsed_pitch_trials_test.mat','file')
  make_parsed_pitch_trials(perttrial_pre,perttrial_dur,fs,ystep,ntrials,pitch_pert,pitch_out,pitch_in,ampl_out,ampl_in,wave_out,wave_in, ...
                           pitchlimits,reject_pitchlimits,parse_fig_pos);
end
load('parsed_pitch_trials_test');
% if length(good_trials) ~= ntrials, error('length(%d) of good_trials ~= ntrials(%d)',length(good_trials),ntrials); end
pert_resp = get_pitch_pert_response2(trial_pert_types,parsed_pitch_in,parsed_pitch_out,good_trials,parsed_frame_taxis,[],pertclass_pitch_types);
%pert_resp = get_pitch_pert_response2(trial_pert_types,parsed_pitch_in,parsed_pitch_out,good_trials,parsed_frame_taxis,pertclass_pitch_types);

if (yes_overwrite >= 3) || ~exist('baseline4comp.mat','file')
  make_baseline4comp(pert_resp,colors,parse_fig_pos);
end
load('baseline4comp');

pert_resp.cents4comp.baseline = baseline;
pert_resp.cents4comp.baseline_type = baseline_type;
npert_types = pert_resp.npert_types;
for ipert_type = 1:(npert_types+1)
  baseline_repmat = repmat(pert_resp.cents4comp.baseline,[pert_resp.n_good_trials(ipert_type) 1]);

  pert_resp.cents4comp.pitch_in.dat{ipert_type} = pert_resp.cents.pitch_in.dat{ipert_type} - baseline_repmat;
  pert_resp.cents4comp.pitch_in.mean(ipert_type,:) = mean(pert_resp.cents4comp.pitch_in.dat{ipert_type},1);
  pert_resp.cents4comp.pitch_in.stde(ipert_type,:) = std(pert_resp.cents4comp.pitch_in.dat{ipert_type},0,1)/sqrt(pert_resp.n_good_trials(ipert_type));
  pert_resp.cents4comp.pitch_out.dat{ipert_type} = pert_resp.cents.pitch_out.dat{ipert_type} - baseline_repmat;
  pert_resp.cents4comp.pitch_out.mean(ipert_type,:) = mean(pert_resp.cents4comp.pitch_out.dat{ipert_type},1);
  pert_resp.cents4comp.pitch_out.stde(ipert_type,:) = std(pert_resp.cents4comp.pitch_out.dat{ipert_type},0,1)/sqrt(pert_resp.n_good_trials(ipert_type));

end

yes_plot_this = 0;
if yes_plot_this
hf = my_figure;
set(hf,'Position',parse_fig_pos);
subplot(211)
plot(parsed_frame_taxis,pert_resp.cents.pitch_in.mean(3,:),'k')
hold on
plot(parsed_frame_taxis,pert_resp.cents.pitch_in.mean(2,:),'r')
plot(parsed_frame_taxis,pert_resp.cents.pitch_in.mean(1,:),'b')
subplot(212)
plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.mean(3,:),'k')
hold on
plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.mean(2,:),'r')
plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.mean(1,:),'b')
end

if (yes_overwrite >= 2) || ~exist('tlims4comp.mat','file')
  make_tlims4comp2(pert_resp,colors,parse_fig_pos);
end
load('tlims4comp');

if (yes_overwrite >= 1) || ~exist('pert_resp.mat','file')
  for ipert_type = 1:npert_types
    pert_resp.mean_signed_response{ipert_type} = mean(pert_resp.cents4comp.pitch_in.dat{ipert_type}(:,idxes4comp),2);
    pert_resp.comp{ipert_type} = -(pert_resp.mean_signed_response{ipert_type})/pert_resp.pert_types(ipert_type);
  end
  tflat_rest_of_resp = [];
  yes_peakpick = 0;
  xcspec = get_pert_resp_xcspec(pert_resp,[],tlims4baseline,[],1,tflat_rest_of_resp,yes_peakpick);
  npert_types = pert_resp.npert_types;
  for ipert_type = 1:npert_types
    pert_resp.peakresp{ipert_type} = xcspec(ipert_type).peakresp';
    pert_resp.peakcomp{ipert_type} = xcspec(ipert_type).peakcomp';
    pert_resp.tonset{ipert_type} = xcspec(ipert_type).tonset';
    pert_resp.tpeak{ipert_type} = xcspec(ipert_type).tpeak';
    pert_resp.xcspec_mean{ipert_type} = xcspec(ipert_type).mean;
  end
  save('pert_resp','pert_resp');
end
load('pert_resp');

if yes_add_ampl_info
  dbtarg = 80;
  for ipert_type = 1:(pert_resp.npert_types + 1)
    n_good_trials = pert_resp.n_good_trials(ipert_type);
    pert_resp.dB_ampl_in.dat{ipert_type} = zeros(n_good_trials,pert_resp.nframeswin);
    pert_resp.dB_ampl_out.dat{ipert_type} = zeros(n_good_trials,pert_resp.nframeswin);
    for iitr = 1:n_good_trials
      itr = pert_resp.good_trials{ipert_type}(iitr);
      ampl_in4resp(iitr,:)  = ampl_in( itr,(iframe_low:iframe_hi) - iframe4pert + i_onsets{itr}(1));
      ampl_out4resp(iitr,:) = ampl_out(itr,(iframe_low:iframe_hi) - iframe4pert + i_onsets{itr}(1));
    end
    dbcorr_in  = dbtarg - 20*log10(median(mean(ampl_in4resp, 2)));
    dbcorr_out = dbtarg - 20*log10(median(mean(ampl_out4resp,2)));
    for iitr = 1:n_good_trials
      pert_resp.dB_ampl_in.dat{ ipert_type}(iitr,:) = 20*log10(ampl_in4resp( iitr,:)) + dbcorr_in;
      pert_resp.dB_ampl_out.dat{ipert_type}(iitr,:) = 20*log10(ampl_out4resp(iitr,:)) + dbcorr_out;
    end
    pert_resp.dB_ampl_in.mean( ipert_type,:) = mean(pert_resp.dB_ampl_in.dat{ ipert_type},  1);
    pert_resp.dB_ampl_in.stde( ipert_type,:) = std( pert_resp.dB_ampl_in.dat{ ipert_type},0,1)/sqrt(n_good_trials);
    pert_resp.dB_ampl_out.mean(ipert_type,:) = mean(pert_resp.dB_ampl_out.dat{ipert_type},  1);
    pert_resp.dB_ampl_out.stde(ipert_type,:) = std( pert_resp.dB_ampl_out.dat{ipert_type},0,1)/sqrt(n_good_trials);
  end
  save('pert_resp','pert_resp');
end

for ipert_type = 1:(npert_types+1)
  baseline_repmat = repmat(pert_resp.cents4comp.baseline,[pert_resp.n_good_trials(ipert_type) 1]);
  pert_resp.cents4comp.pitch_in.t(ipert_type,:) = ttest(pert_resp.cents4comp.pitch_in.dat{ipert_type});
  pert_resp.cents4comp.pitch_in.median(ipert_type,:) = median(pert_resp.cents4comp.pitch_in.dat{ipert_type},1);
  pert_resp.cents4comp.pitch_out.median(ipert_type,:) = median(pert_resp.cents4comp.pitch_out.dat{ipert_type},1);
end

allcomp = [];
for ipert_type = 1:npert_types
  allcomp = [allcomp; pert_resp.comp{ipert_type}];
end
mean_comp = mean(allcomp);
stde_comp = std(allcomp)/sqrt(length(allcomp));

yes_median4mean = 0;

hf = my_figure;
for iipert_type = 1:npert_types
  % ipert_type = npert_types - iipert_type + 1;
  ipert_type = iipert_type;
  subplot(npert_types,2,2*(iipert_type-1)+1)
  if yes_median4mean
    m = pert_resp.cents4comp.pitch_in.median(ipert_type,:);
  else
    m = pert_resp.cents4comp.pitch_in.mean(ipert_type,:);
  end
  se = pert_resp.cents4comp.pitch_in.stde(ipert_type,:);
  hp = plot(parsed_frame_taxis,m,'r'); set(hp,'LineWidth',3);
  hold on
  hp = plot(parsed_frame_taxis,m-se,'r'); set(hp,'LineWidth',1);
  hp = plot(parsed_frame_taxis,m+se,'r'); set(hp,'LineWidth',1);

  if yes_median4mean
    m = pert_resp.cents4comp.pitch_out.median(ipert_type,:);
  else
    m = pert_resp.cents4comp.pitch_out.mean(ipert_type,:);
  end
  se = pert_resp.cents4comp.pitch_out.stde(ipert_type,:);
  hp = plot(parsed_frame_taxis,m,'b'); set(hp,'LineWidth',3);
  hp = plot(parsed_frame_taxis,m-se,'b'); set(hp,'LineWidth',1);
  hp = plot(parsed_frame_taxis,m+se,'b'); set(hp,'LineWidth',1);
  
  hvp = vpatch(0,0.4,'g');
  set(hvp,'FaceAlpha',0.25);
  
  if yes_median4mean
    m = pert_resp.cents4comp.pitch_in.median(ipert_type,:);
  else
    m = pert_resp.cents4comp.pitch_in.mean(ipert_type,:);
  end
  se = pert_resp.cents4comp.pitch_in.stde(ipert_type,:);
  hp = plot(parsed_frame_taxis,m,'r'); set(hp,'LineWidth',3);
  hold on
  hp = plot(parsed_frame_taxis,m-se,'r'); set(hp,'LineWidth',1);
  hp = plot(parsed_frame_taxis,m+se,'r'); set(hp,'LineWidth',1);

  if yes_median4mean
    m = pert_resp.cents4comp.pitch_out.median(ipert_type,:);
  else
    m = pert_resp.cents4comp.pitch_out.mean(ipert_type,:);
  end
  se = pert_resp.cents4comp.pitch_out.stde(ipert_type,:);
  hp = plot(parsed_frame_taxis,m,'b'); set(hp,'LineWidth',3);
  hp = plot(parsed_frame_taxis,m-se,'b'); set(hp,'LineWidth',1);
  hp = plot(parsed_frame_taxis,m+se,'b'); set(hp,'LineWidth',1);
  
  hold off
  if ipert_type == 1
    axis([parsed_frame_taxis(1) parsed_frame_taxis(end) -100 250]);
  else
    axis([parsed_frame_taxis(1) parsed_frame_taxis(end) -250 100]);
  end
  xlabel('time (sec)');
  ylabel('corr. cents');
  
  subplot(npert_types,2,2*(iipert_type-1)+2)
  if yes_median4mean
    m = pert_resp.cents4comp.pitch_in.median(ipert_type,:);
  else
    m = pert_resp.cents4comp.pitch_in.mean(ipert_type,:);
  end
  se = pert_resp.cents4comp.pitch_in.stde(ipert_type,:);
  hp = plot(parsed_frame_taxis,m,'k'); set(hp,'LineWidth',3);
  hold on
  hp = plot(parsed_frame_taxis,m-se,'k'); set(hp,'LineWidth',1);
  hp = plot(parsed_frame_taxis,m+se,'k'); set(hp,'LineWidth',1);
  
  plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.dat{ipert_type}');
  
  hvp = vpatch(0,0.4,'g');
  set(hvp,'FaceAlpha',0.25);
  
  hp = plot(parsed_frame_taxis,m,'k'); set(hp,'LineWidth',3);
  hp = plot(parsed_frame_taxis,m-se,'k'); set(hp,'LineWidth',1);
  hp = plot(parsed_frame_taxis,m+se,'k'); set(hp,'LineWidth',1);

  plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.dat{ipert_type}');
  
  hold off
  axis([parsed_frame_taxis(1) parsed_frame_taxis(end) -150 150]);
  xlabel('time (sec)');
  ylabel('corr. cents');
end

nbins = 20;
[bincounts_comp,bincens_comp] = hist(allcomp,nbins);
[bincounts_pos_comp,bincens_pos_comp] = hist(pert_resp.comp{1},nbins); mean_pos_comp = mean(pert_resp.comp{1});
[bincounts_neg_comp,bincens_neg_comp] = hist(pert_resp.comp{2},nbins); mean_neg_comp = mean(pert_resp.comp{2});
hf = my_figure;
subplot(311)
hb = bar(bincens_comp,bincounts_comp);
a = axis; axis([-1 1 a(3:4)]);
subplot(312)
hb = bar(bincens_pos_comp,bincounts_pos_comp);
axis([-1 1 a(3:4)]);
hl = vline(0,'r'); set(hl,'LineWidth',2);
ylabel('pos'); xlabel('comp');
hl = vline(mean_pos_comp,'g');
subplot(313)
hb = bar(bincens_neg_comp,bincounts_neg_comp);
axis([-1 1 a(3:4)]);
hl = vline(0,'r'); set(hl,'LineWidth',2);
ylabel('neg'); xlabel('comp');
hl = vline(mean_neg_comp,'g');
subplot(311)
hb = bar(bincens_comp,bincounts_comp);
axis([-1 1 a(3:4)]);
hl = vline(0,'r'); set(hl,'LineWidth',2);
ylabel('all'); xlabel('comp');
hl = vline(mean_comp,'g');
plotname = sprintf('subj%d_pitch_comp_hist',isubj);
plot_title = sprintf('%s,      mean_comp(%.4f), stde_comp(%.4f)',plotname,mean_comp,stde_comp);
fprintf('%s\n',plot_title);
ht = title(plot_title);
set(ht,'Interpreter','none');
set(gcf,'Name',plotname);

yes_plot_trials2print = 0;
if yes_plot_trials2print
  
  % below this point is code for displaying details of each good trial of the experiment

  yes_play_trial = 0;
  yes_show_wave_in = 0;
  yes_show_gram_in = 0;
  yes_show_wave_out = 0;
  yes_show_gram_out = 0;
  yes_show_hz = 0;
  yes_show_cents = 1;
  yes_show_pert = 0;
  yes_show_comp_hist = 1;
  
  yes_log_gram = 1;
  
  fig_bord = 0.15; subplot_topbord = 0.1; subplot_rightbord = -0.1;
  
  if yes_show_gram_in || yes_show_gram_out
    % ms_window = 36;
    ms_window = 72;
    nfft = 2*2*4096;
    nsamp_window = round(ms_window*fs/1000);
    nsamp_frame_advance = ystep
    nsamp_overlap = nsamp_window - nsamp_frame_advance;
    Flims = [0 570];
  end
  
  hf = my_figure;
  % set(hf,'Position',[190     7   560   722]);
  % set(hf,'Position',[148   132   890   722]);
  % set(hf,'Position',  [148   132   890  1214]);
  set(hf,'Position',  [1311         550         391         919]);
  set(hf,'PaperPositionMode','auto');
  nsubfigs = 1;
  if yes_show_comp_hist, nsubfigs = nsubfigs + 1; end
  if yes_show_wave_in, nsubfigs = nsubfigs + 1; end
  if yes_show_gram_in, nsubfigs = nsubfigs + 1; end
  if yes_show_wave_out, nsubfigs = nsubfigs + 1; end
  if yes_show_gram_out, nsubfigs = nsubfigs + 1; end
  if yes_show_hz, nsubfigs = nsubfigs + 1; end
  
  cd(curdir_subj)
  for iipert_type = npert_types:(-1):1
    comps4pert =         pert_resp.comp{iipert_type};
    n_good_trials4pert = pert_resp.n_good_trials(iipert_type);
    good_trials4pert =   pert_resp.good_trials{iipert_type};
    i_onsets4pert =                [i_onsets{good_trials4pert}];
    pitchs_in4pert =                pitch_in(good_trials4pert,:);
    pitchs_out4pert =              pitch_out(good_trials4pert,:);
    pitch_perts4pert =            pitch_pert(good_trials4pert,:);
    itrials_in_blocks4pert = itrial_in_block(good_trials4pert);
    iblocks_in_trials4pert =          iblock(good_trials4pert);
    iblocks4pert = unique(iblocks_in_trials4pert);
    nblocks4pert = length(iblocks4pert);
    ii4perttrial = 0;
    for iiblock4pert = 1:nblocks4pert
      iblock4pert = iblocks4pert(iiblock4pert);
      expr_audiodir = speak_audiodirs{iblock4pert};
      cd(expr_audiodir); fprintf('cd expr_audiodir(%s)\n',expr_audiodir);
      if yes_play_trial || yes_show_wave_in || yes_show_gram_in, vechist_in  = get_vec_hist6('inbuffer',3); end
      if yes_play_trial || yes_show_wave_out || yes_show_gram_out, vechist_out = get_vec_hist6('outbuffer',3); end
      ntrials_in_block = sum(iblocks_in_trials4pert == iblock4pert);
      for itr = 1:ntrials_in_block
        ii4perttrial = ii4perttrial + 1;
        itrial_in_block4pert = itrials_in_blocks4pert(ii4perttrial);
        i_onset4trial = i_onsets4pert(ii4perttrial);
        i_winstart = i_onset4trial - nframes_pre; if i_winstart < 1, error(sprintf('i_winstart(%d) < 1',i_winstart)); end
        i_winstop = i_winstart + nframeswin - 1; if i_winstop > nframes, error(sprintf('i_winstop(%d) > nframes(%d)',i_winstop,nframes)); end
        isamp_idxes = ((i_winstart-1)*ystep + 1):(i_winstop*ystep);
        pert_taxis = (isamp_idxes - i_onset4trial*ystep)/fs;
        pert_taxis_frame = ((i_winstart:i_winstop)-i_onset4trial)*ystep/fs;
        pitch_in4trial =   pitchs_in4pert(ii4perttrial,:);
        pitch_out4trial = pitchs_out4pert(ii4perttrial,:);
        pert_pitch_cents_in4trial = pert_resp.cents4comp.pitch_in.dat{iipert_type}(ii4perttrial,:);
        pert_pitch_cents_out4trial = pert_resp.cents4comp.pitch_out.dat{iipert_type}(ii4perttrial,:);
        pitch_pert4trial = pitch_perts4pert(ii4perttrial,i_winstart:i_winstop);
        comp4trial = comps4pert(ii4perttrial);
        clf
        isubfig = 0;
        if yes_show_wave_in, isubfig = isubfig + 1; hax(isubfig) = my_subplot(nsubfigs,1,isubfig,fig_bord,subplot_topbord,subplot_rightbord);
          ywave = getwave_vec_hist6(vechist_in,itrial_in_block4pert,fs); ywave_seg = ywave(isamp_idxes);
          plot(pert_taxis,ywave_seg);
          set(gca,'XTickLabel',[]);
          ylabel('ampl');
          a = axis; axis([pert_taxis(1) pert_taxis(end) a(3:4)]);
        end
        if yes_show_gram_in, isubfig = isubfig + 1; hax(isubfig) = my_subplot(nsubfigs,1,isubfig,fig_bord,subplot_topbord,subplot_rightbord);
          ywave = getwave_vec_hist6(vechist_in,itrial_in_block4pert,fs);
          [S,F,T] = spectrogram(ywave,nsamp_window,nsamp_overlap,nfft,fs);
          T = T - i_onset4trial*ystep/fs;
          iFlims = dsearchn(F,Flims')';
          S = S(iFlims(1):iFlims(2),:);
          absS = abs(S);
          if yes_log_gram
            hgram_in = pcolor(T,F(iFlims(1):iFlims(2)),20*log10(absS));
          else
            hgram_in = pcolor(T,F(iFlims(1):iFlims(2)),absS);
          end
          shading interp
          set(gca,'Layer','top');
          set(gca,'XTickLabel',[]);
          ylabel('freq (Hz)');
          a = axis; axis([pert_taxis(1) pert_taxis(end) a(3:4)]);
        end
        if yes_show_wave_out, isubfig = isubfig + 1; hax(isubfig) = my_subplot(nsubfigs,1,isubfig,fig_bord,subplot_topbord,subplot_rightbord);
          ywave = getwave_vec_hist6(vechist_out,itrial_in_block4pert,fs); ywave_seg = ywave(isamp_idxes);
          plot(pert_taxis,ywave_seg);
          set(gca,'XTickLabel',[]);
          ylabel('ampl');
          a = axis; axis([pert_taxis(1) pert_taxis(end) a(3:4)]);
        end
        if yes_show_gram_out, isubfig = isubfig + 1; hax(isubfig) = my_subplot(nsubfigs,1,isubfig,fig_bord,subplot_topbord,subplot_rightbord);
          ywave = getwave_vec_hist6(vechist_out,itrial_in_block4pert,fs);
          [S,F,T] = spectrogram(ywave,nsamp_window,nsamp_overlap,nfft,fs);
          T = T - i_onset4trial*ystep/fs;
          iFlims = dsearchn(F,Flims')';
          S = S(iFlims(1):iFlims(2),:);
          absS = abs(S);
          if yes_log_gram
            hgram_out = pcolor(T,F(iFlims(1):iFlims(2)),20*log10(absS));
          else
            hgram_out = pcolor(T,F(iFlims(1):iFlims(2)),absS);
          end
          shading interp
          set(gca,'Layer','top');
          set(gca,'XTickLabel',[]);
          ylabel('freq (Hz)');
          a = axis; axis([pert_taxis(1) pert_taxis(end) a(3:4)]);
        end
        if yes_show_hz, isubfig = isubfig + 1; hax(isubfig) = my_subplot(nsubfigs,1,isubfig,fig_bord,subplot_topbord,subplot_rightbord);
          plot(pert_taxis_frame,squeeze(pitch_out4trial(i_winstart:i_winstop)),'b'); hold on
          plot(pert_taxis_frame,squeeze(pitch_in4trial(i_winstart:i_winstop)),'r');  hold off
          set(gca,'XTickLabel',[]);
          ylabel('pitch (Hz)');
          a = axis; axis([pert_taxis_frame(1) pert_taxis_frame(end) a(3:4)]);
        end
        if yes_show_cents, isubfig = isubfig + 1; hax(isubfig) = my_subplot(nsubfigs,1,isubfig,fig_bord,subplot_topbord,subplot_rightbord);
          hpl = plot(pert_taxis_frame,pert_pitch_cents_out4trial,'b'); set(hpl,'LineWidth',3); hold on
          hpl = plot(pert_taxis_frame,pert_pitch_cents_in4trial,'r'); set(hpl,'LineWidth',3); 
          if yes_show_pert, hpl = plot(pert_taxis_frame,pitch_pert4trial,'g'); set(hpl,'LineWidth',3); end
          hold off
          xlabel('time (sec)');
          ylabel('pitch (cents)');
          a = axis; axis([pert_taxis_frame(1) pert_taxis_frame(end) a(3:4)]);
        end
        if yes_show_comp_hist, isubfig = isubfig + 1; hax(isubfig) = my_subplot(nsubfigs,1,isubfig,fig_bord,subplot_topbord,subplot_rightbord);
          hb = bar(bincens_comp,bincounts_comp); hold on
          set(hb,'EdgeColor','none')
          set(hb,'FaceColor','r')
          hold off
          axis([-1 1 0 1.05*max(bincounts_comp)]);
          hl = vline(0,'k'); set(hl,'LineWidth',1.5);
          ylabel('# trials'); xlabel('comp');
          ibin_of_comp = dsearchn(bincens_comp',comp4trial);
          bincount_of_comp = bincounts_comp(ibin_of_comp);
          % add an annonation arrow to the comp histogram
          a = axis;
          arrow_axfract_xpos = (comp4trial - a(1))/(a(2) - a(1));
          arrow_axfract_ypos = (bincount_of_comp - a(3))/(a(4) - a(3));
          arrow_axfract_len = 0.1;
          axpos = get(gca,'Position');
          arrow_figfract_xpos(1) = axpos(1) + arrow_axfract_xpos*axpos(3);
          arrow_figfract_xpos(2) = axpos(1) + arrow_axfract_xpos*axpos(3);
          arrow_figfract_ypos(1) = axpos(2) + (arrow_axfract_ypos+arrow_axfract_len)*axpos(4);
          arrow_figfract_ypos(2) = axpos(2) + arrow_axfract_ypos*axpos(4);
          harrow = annotation('arrow',arrow_figfract_xpos,arrow_figfract_ypos);
          set(harrow,'Color','g')
          set(harrow,'LineWidth',2)
        end
        ht = title(hax(1),sprintf('%.f cents pert,igoodtrial(%d): iblock(%d),itrial_in_block(%d): comp(%.3f)', ...
                                  pert_resp.pert_types(iipert_type),ii4perttrial,iblock4pert,itrial_in_block4pert,comp4trial));
        set(ht,'Interpreter','none');
        axpos = get(hax(2),'Position');
        set(hax(2),'Position',[axpos(1) axpos(2)+0.02 axpos(3:4)]);
        reply = input('print fig? y/[n]: ','s');
        if strcmp(reply,'y')
          if exist('hgram_in','var') || exist('hgram_out','var')
            print -dtiff Fig1bc.tiff
          end
          if exist('hgram_out','var'), delete(hgram_out); end
          if exist('hgram_in', 'var'), delete(hgram_in); end
          print -depsc Fig1bc.eps
        end
      end
      cd(curdir_subj);
    end
  end
end % if yes_plot_trials2print

yes_plot_corrcomp = 0;
if yes_plot_corrcomp
  % i_itvl4corr = [1 (iframe4pert-1)];
  % i_itvl4corr = [iframe4pert nframeswin];
  i_itvl4corr = [(iframe4pert-50) (iframe4pert+50)];
  
  yes_plot_trialnum_comp_corr = 1;
  yes_plot_pitch_comp_corr = 1;
  yes_plot_ampl_comp_corr = 1;
  
  hf_trialnum = figure; for i = 1:3, trialnum_hax(i) = subplot(3,1,i); end
  hf_corrcomp = figure; for i = 1:3, corrcomp_hax(i) = subplot(3,1,i); end
  
  set(hf_trialnum,'Position',[1000         266         560        1231]);
  set(hf_corrcomp,'Position',[1582         266         560        1231]);
  
  if yes_plot_trialnum_comp_corr
    trial_num_dat = [pert_resp.good_trials{1} pert_resp.good_trials{2}]';
    comp_trial_dat = [(pert_resp.comp{1});(pert_resp.comp{2})];
    [b,duh,duh1,duh2,stats] = regress(comp_trial_dat,[ones(size(trial_num_dat)) trial_num_dat]);
    r_trialnum_comp = sqrt(stats(1));
    p_trialnum_comp = stats(3);
    axes(corrcomp_hax(1));
    plot(trial_num_dat,comp_trial_dat,'*');
    hrl = refline(b(2),b(1));
    set(hrl,'Color','r');
    set(hrl,'LineWidth',2);
    xlabel('trial #');
    ylabel('comp');
    title(sprintf('subj(%d), r(%.2f), p(%.5f)',isubj,r_trialnum_comp,p_trialnum_comp));
    axes(trialnum_hax(1));
    plot(trial_num_dat,comp_trial_dat,'*');
    set(hrl,'Color','r');
    set(hrl,'LineWidth',2);
    xlabel('trial #');
    ylabel('comp');
    title(sprintf('subj(%d)',isubj));
  end
  
  if yes_plot_pitch_comp_corr
    pitch_trial_dat = mean([(pert_resp.pitch_in.dat{1}(:,i_itvl4corr(1):i_itvl4corr(2))); ...
                        (pert_resp.pitch_in.dat{2}(:,i_itvl4corr(1):i_itvl4corr(2)))],2);
    comp_trial_dat = [(pert_resp.comp{1});(pert_resp.comp{2})];
    [b,duh,duh1,duh2,stats] = regress(comp_trial_dat,[ones(size(pitch_trial_dat)) pitch_trial_dat]);
    r_pitch_comp = sqrt(stats(1));
    p_pitch_comp = stats(3);
    axes(corrcomp_hax(2));
    plot(pitch_trial_dat,comp_trial_dat,'*');
    hrl = refline(b(2),b(1));
    set(hrl,'Color','r');
    set(hrl,'LineWidth',2);
    xlabel('pitch (Hz)');
    ylabel('comp');
    title(sprintf('subj(%d), r(%.2f), p(%.5f)',isubj,r_pitch_comp,p_pitch_comp));
    axes(trialnum_hax(2));
    plot([pert_resp.good_trials{1} pert_resp.good_trials{2}],pitch_trial_dat,'*');
    xlabel('trial #');
    ylabel('pitch (Hz)');
    title(sprintf('subj(%d)',isubj));
  end
  
  if yes_plot_ampl_comp_corr
    ampl_trial_dat = mean([(pert_resp.dB_ampl_in.dat{1}(:,i_itvl4corr(1):i_itvl4corr(2))); ...
                        (pert_resp.dB_ampl_in.dat{2}(:,i_itvl4corr(1):i_itvl4corr(2)))],2);
    comp_trial_dat = [(pert_resp.comp{1});(pert_resp.comp{2})];
    [b,duh,duh1,duh2,stats] = regress(comp_trial_dat,[ones(size(ampl_trial_dat)) ampl_trial_dat]);
    r_ampl_comp = sqrt(stats(1));
    p_ampl_comp = stats(3);
    axes(corrcomp_hax(3));
    plot(ampl_trial_dat,comp_trial_dat,'*');
    hrl = refline(b(2),b(1));
    set(hrl,'Color','r');
    set(hrl,'LineWidth',2);
    xlabel('ampl (dB)');
    ylabel('comp');
    title(sprintf('subj(%d), r(%.2f), p(%.5f)',isubj,r_ampl_comp,p_ampl_comp));
    axes(trialnum_hax(3));
    plot([pert_resp.good_trials{1} pert_resp.good_trials{2}],ampl_trial_dat,'*');
    xlabel('trial #');
    ylabel('ampl (Hz)');
    title(sprintf('subj(%d)',isubj));
  end
  
  figure(hf_trialnum);
  set(hf_trialnum,'PaperPositionMode','auto');
  print('-depsc',sprintf('subj%d_trialpitchampl_trialnum',isubj));
  figure(hf_corrcomp);
  set(hf_corrcomp,'PaperPositionMode','auto');
  print('-depsc',sprintf('subj%d_trialpitchampl_corrcomp',isubj));
end

cd(curdir)

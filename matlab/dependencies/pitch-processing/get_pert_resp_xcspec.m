function xcspec = get_pert_resp_xcspec(pert_resp,tlims4pert,tlims4baseline,min_ms_beyond_th,yes_plot,tflat_rest_of_resp,yes_peakpick)

if nargin < 2 || isempty(tlims4pert), tlims4pert = [0 0.4]; end
if nargin < 3 || isempty(tlims4baseline), tlims4baseline = [pert_resp.frame_taxis(1) 0]; end
if nargin < 4 || isempty(min_ms_beyond_th), min_ms_beyond_th = 50; end
if nargin < 5 || isempty(yes_plot), yes_plot = 1; end
if nargin < 6, tflat_rest_of_resp = []; end
if nargin < 7 || isempty(yes_peakpick), yes_peakpick = 0; end

if ~isempty(tflat_rest_of_resp)
  iflat_rest_of_resp = dsearchn(pert_resp.frame_taxis',tflat_rest_of_resp);
else
  iflat_rest_of_resp = [];
end

% xcorr_opt = 'none';
xcorr_opt = 'unbiased';
% xcorr_opt = 'coeff';

frame_fs = 1/(mean(diff(pert_resp.frame_taxis)));

ilims4pert = dsearchn(pert_resp.frame_taxis',tlims4pert')';
ilims4baseline = dsearchn(pert_resp.frame_taxis',tlims4baseline')';
min_iframes_beyond_th = round((min_ms_beyond_th/1000)*frame_fs);

if yes_plot
  h_mean_resp_fig = figure;
end

if yes_peakpick
  hf_peakpick = figure;
end

npert_types = pert_resp.npert_types;
for ipert_type = 1:npert_types

  mean_resp = pert_resp.cents4comp.pitch_in.mean(ipert_type,:);

  mean_resp(1:ilims4baseline(1)) = 0;
  if ~isempty(iflat_rest_of_resp) % what a hack!
    mean_resp(iflat_rest_of_resp:end) = mean_resp(iflat_rest_of_resp);
  end

  base_mean_resp = mean_resp(ilims4baseline(1):ilims4baseline(2));
  mean_base_mean_resp = mean(base_mean_resp);
  stdv_base_mean_resp = std(base_mean_resp);
  sign_of_resp = -sign(pert_resp.pert_types(ipert_type));
  response_onset_thresh = mean_base_mean_resp + 2*sign_of_resp*stdv_base_mean_resp;
  
  if sign_of_resp > 0
    iframes_mean_resp_exceed_th = find(mean_resp(ilims4baseline(2):end) > response_onset_thresh) + ilims4baseline(2) - 1;
  else
    iframes_mean_resp_exceed_th = find(mean_resp(ilims4baseline(2):end) < response_onset_thresh) + ilims4baseline(2) - 1;
  end
  nframes_mean_resp_exceed_th = length(iframes_mean_resp_exceed_th);
  iiframe_beyond_th_discon = find(diff([-1 iframes_mean_resp_exceed_th]) > 1);
  lengths_after_discon = diff([iiframe_beyond_th_discon (nframes_mean_resp_exceed_th+1)]);
  niiframes_beyond_th_discon = length(iiframe_beyond_th_discon);
  valid_onset_found = 0;
  for iiiframe = 1:niiframes_beyond_th_discon
    if lengths_after_discon(iiiframe) > min_iframes_beyond_th
      valid_onset_found = 1;
      break;
    end
  end
  if valid_onset_found
    ionset_mean_resp = iframes_mean_resp_exceed_th(iiframe_beyond_th_discon(iiiframe));
  else
    ionset_mean_resp = mean(ilims4pert);
  end
  
  [maxpeak_mean_resp, imaxpeak_mean_resp] = max(sign_of_resp*mean_resp(ilims4baseline(2):end));
  ipeak_mean_resp = imaxpeak_mean_resp + ilims4baseline(2) - 1;
  peak_mean_resp = mean_resp(ipeak_mean_resp);
  mean_peakcomp = -peak_mean_resp/(pert_resp.pert_types(ipert_type));

  if yes_plot
    figure(h_mean_resp_fig);
    subplot(npert_types,1,ipert_type)
    plot(mean_resp)
    hl = vline(ilims4pert(1),'k');
    hl = vline(ilims4pert(2),'k');
    hl = hline(response_onset_thresh,'k');
    hl_ionset = vline(ionset_mean_resp,'g');
    hl_ipeak = vline(ipeak_mean_resp,'r');
    hl_peakampl = hline(peak_mean_resp,'r');
    ht = title(sprintf('ipert(%d): peak_signed_response(%f) peak_comp(%f)',ipert_type,peak_mean_resp,mean_peakcomp)); set(ht,'Interpreter','none');
    while 1
      reply = input('happy with response onset? [y]/n: ','s');
      if isempty(reply) || strcmp(reply,'y'), break; end
      fprintf('pick response onset\n');
      [xx,yy] = ginput(1);
      ionset_mean_resp = round(xx);
      set(hl_ionset,'XData',ionset_mean_resp*[1 1]);
    end
    while 1
      reply = input('happy with response peak? [y]/n: ','s');
      if isempty(reply) || strcmp(reply,'y'), break; end
      fprintf('pick response peak\n');
      [xx,yy] = ginput(1);
      ipeak_mean_resp = round(xx);
      peak_mean_resp = mean_resp(ipeak_mean_resp);
      mean_peakcomp = -peak_mean_resp/(pert_resp.pert_types(ipert_type));
      set(hl_ipeak,'XData',ipeak_mean_resp*[1 1]);
      set(hl_peakampl,'YData',peak_mean_resp*[1 1]);
    end
  end
  
  % a valid given response can't start before the pert onset,
  % so we'll also say anything lagging the mean more than this is not believed
  min_rel_lag = -(ipeak_mean_resp - ilims4pert(1));
  max_rel_lag =  (length(pert_resp.frame_taxis) - ipeak_mean_resp);

  ac = xcorr(mean_resp,mean_resp,xcorr_opt);
  [mac,imac] = max(abs(ac));
  
  xcspec(ipert_type).mean.resp = mean_resp;
  xcspec(ipert_type).mean.resp_onset_th = response_onset_thresh;
  xcspec(ipert_type).mean.ionset = ionset_mean_resp;
  xcspec(ipert_type).mean.ipeak = ipeak_mean_resp;
  xcspec(ipert_type).mean.peak = peak_mean_resp;
  xcspec(ipert_type).mean.peakcomp = mean_peakcomp;
  xcspec(ipert_type).mean.ac = ac;
  xcspec(ipert_type).mean.mac = mac;
  xcspec(ipert_type).mean.imac = imac;
  xcspec(ipert_type).mean.min_rel_lag = min_rel_lag;
  xcspec(ipert_type).mean.max_rel_lag = max_rel_lag;
  
  ntrials = pert_resp.n_good_trials(ipert_type);
  for itrial = 1:ntrials
    trial_resp = pert_resp.cents4comp.pitch_in.dat{ipert_type}(itrial,:);

    trial_resp(1:ilims4baseline(1)) = 0;
    if ~isempty(iflat_rest_of_resp)
      trial_resp(iflat_rest_of_resp:end) = trial_resp(iflat_rest_of_resp);
    end

    trial_resp_baseline_mean = mean(trial_resp(ilims4baseline(1):ilims4baseline(2)));
    trial_resp_baseline_stdv = std(trial_resp(ilims4baseline(1):ilims4baseline(2)));
    cen_trial_resp = trial_resp - trial_resp_baseline_mean;
    the_xc = xcorr(cen_trial_resp,mean_resp,xcorr_opt);

    [ipospxc,apospxc,npospxc] = peakfind(the_xc);
    posp_rel_lags = ipospxc - imac;
    ii_valid_posp_rel_lags = find((posp_rel_lags > min_rel_lag) & (posp_rel_lags < max_rel_lag));
    iposp_valid_lags = ipospxc(ii_valid_posp_rel_lags);
    aposp_valid_lags = apospxc(ii_valid_posp_rel_lags);
    [maposp,imaposp] = max(abs(aposp_valid_lags));
    ibest_posp_lag = iposp_valid_lags(imaposp);
    abest_posp_lag = aposp_valid_lags(imaposp);
    
    [inegpxc,anegpxc,nnegpxc] = peakfind(-the_xc);
    negp_rel_lags = inegpxc - imac;
    ii_valid_negp_rel_lags = find((negp_rel_lags > min_rel_lag) & (negp_rel_lags < max_rel_lag));
    inegp_valid_lags = inegpxc(ii_valid_negp_rel_lags);
    anegp_valid_lags = anegpxc(ii_valid_negp_rel_lags);
    [manegp,imanegp] = max(abs(anegp_valid_lags));
    ibest_negp_lag = inegp_valid_lags(imanegp);
    abest_negp_lag = anegp_valid_lags(imanegp);
    
    
    switch (10*(~isempty(ibest_posp_lag)) + (~isempty(ibest_negp_lag)))
      case 11, if abest_posp_lag >= abest_negp_lag, choose_posp_lag = 1; else, choose_posp_lag = 0; end
      case 10, choose_posp_lag = 1;
      case 01, choose_posp_lag = 0;
      case 00, error('no valid lag to choose from');
    end
    if choose_posp_lag
      the_ibest_lag = ibest_posp_lag; the_abest_lag =  abest_posp_lag;
    else
      the_ibest_lag = ibest_negp_lag; the_abest_lag = -abest_negp_lag;
    end      

    if yes_peakpick
      figure(hf_peakpick)
      clf
      subplot(212);
      plot(the_xc);
      vline(xcspec(ipert_type).mean.imac,'k');
      hlbest_xc = vline(the_ibest_lag,'y'); set(hlbest_xc,'LineWidth',3);
      for i = 1:length(iposp_valid_lags), hl(i) = vline( iposp_valid_lags(i),'r'); end
      for i = 1:length(inegp_valid_lags), hl(i) = vline( inegp_valid_lags(i),'g'); end
      subplot(211);
      hpl = plot(cen_trial_resp);
      ht = title(sprintf('ipert_type(%d) itrial(%d)',ipert_type,itrial)); set(ht,'Interpreter','none');
      hold on
      hpl = plot(mean_resp,'m');
      vline(xcspec(ipert_type).mean.ipeak,'k');
      lag_ref = xcspec(ipert_type).mean.ipeak - xcspec(ipert_type).mean.imac;
      hlbest = vline(the_ibest_lag + lag_ref,'y'); set(hlbest,'LineWidth',3);
      for i = 1:length(iposp_valid_lags), hl(i) = vline( iposp_valid_lags(i) + lag_ref,'r'); end
      for i = 1:length(inegp_valid_lags), hl(i) = vline( inegp_valid_lags(i) + lag_ref,'g'); end
      a = axis;
      hpa = patch([ilims4pert(1) ilims4pert(2) ilims4pert(2) ilims4pert(1) ilims4pert(1)],[a(3) a(3) a(4) a(4) a(3)],'r');
      set(hpa,'LineStyle','none')
      set(hpa,'FaceColor',0.85*[1 1 1])
      move2back([],hpa);
      while 1
        reply = input('happy with peak pick? [y]/n/(m)anual: ','s');
        if isempty(reply) || strcmp(reply,'y'), break; end
        if strcmp(reply,'n')
          fprintf('pick lag\n');
          [xx,yy] = ginput(1);
          if ~isempty(ibest_posp_lag), [i2pick_pos,d_i2pick_pos] = dsearchn(iposp_valid_lags' + lag_ref,xx); end
          if ~isempty(ibest_negp_lag), [i2pick_neg,d_i2pick_neg] = dsearchn(inegp_valid_lags' + lag_ref,xx); end
          switch (10*(~isempty(ibest_posp_lag)) + (~isempty(ibest_negp_lag)))
            case 11, if d_i2pick_pos <= d_i2pick_neg, choose_d_pos = 1; else, choose_d_pos = 0; end
            case 10, choose_d_pos = 1;
            case 01, choose_d_pos = 0;
            case 00, error('cannot reach this case!');
          end
          if choose_d_pos
            the_ibest_lag = iposp_valid_lags(i2pick_pos); the_abest_lag = aposp_valid_lags(i2pick_pos);
          else
            the_ibest_lag = inegp_valid_lags(i2pick_neg); the_abest_lag = anegp_valid_lags(i2pick_neg);
          end
        else % maunually choose the_ibest_lag (click on a peak in the upper subfig)
          [xx,yy] = ginput(1);
          the_ibest_lag = round(xx - lag_ref);
          the_abest_lag = the_xc(the_ibest_lag);
        end
        set(hlbest,'XData',(the_ibest_lag + lag_ref)*[1 1]);
        set(hlbest_xc,'XData',(the_ibest_lag)*[1 1]);
      end
    end
    
    xcspec(ipert_type).trial_resp_baseline_mean(itrial) = trial_resp_baseline_mean;
    xcspec(ipert_type).trial_resp_baseline_stdv(itrial) = trial_resp_baseline_stdv;
    xcspec(ipert_type).cen_trial_resp(itrial,:) = cen_trial_resp;
    xcspec(ipert_type).xc_resp(itrial,:) = the_xc;
    xcspec(ipert_type).ibest_lag(itrial) = the_ibest_lag;
    xcspec(ipert_type).abest_lag(itrial) = the_abest_lag;
    xcspec(ipert_type).peakresp(itrial) = (the_abest_lag/mac)*peak_mean_resp;
    xcspec(ipert_type).peakcomp(itrial) = xcspec(ipert_type).peakresp(itrial) / (-pert_resp.pert_types(ipert_type));
    xcspec(ipert_type).ipeak(itrial)  =  xcspec(ipert_type).mean.ipeak + xcspec(ipert_type).ibest_lag(itrial) - xcspec(ipert_type).mean.imac;
    xcspec(ipert_type).tpeak(itrial)  =  pert_resp.frame_taxis(xcspec(ipert_type).ipeak(itrial));
    xcspec(ipert_type).ionset(itrial) =  xcspec(ipert_type).mean.ionset + xcspec(ipert_type).ibest_lag(itrial) - xcspec(ipert_type).mean.imac;
    % ionsets are kind of fake: they're simply an extrapolation from the ibest_lag values, and so can have impossible values
    % (i.e., less than ilims4pert(1)). Thus, the next two lines of code keep the values from being impossible.
    if xcspec(ipert_type).ionset(itrial) < 1, xcspec(ipert_type).ionset(itrial) = 1; end 
    xcspec(ipert_type).tonset(itrial) =  threshspec(pert_resp.frame_taxis(xcspec(ipert_type).ionset(itrial)),0);
  end
end
if yes_peakpick
  delete(hf_peakpick);
end

if yes_plot
  figure
  nbins = 20;
  npert_types = pert_resp.npert_types;
  for ipert_type = 1:npert_types
    pert_min_t(ipert_type) = min([min(xcspec(ipert_type).tonset) min(xcspec(ipert_type).tpeak)]);
    pert_max_t(ipert_type) = max([max(xcspec(ipert_type).tonset) max(xcspec(ipert_type).tpeak)]);
    pert_tonset_bincounts(ipert_type,:) = hist(xcspec(ipert_type).tonset,nbins);
    pert_tpeak_bincounts(ipert_type,:) = hist(xcspec(ipert_type).tpeak,nbins);
  end
  min_t = min(pert_min_t);
  max_t = max(pert_max_t);
  max_bincounts = max([pert_tonset_bincounts(:)' pert_tpeak_bincounts(:)']);
  for ipert_type = 1:npert_types
    isubplot = 0*npert_types + ipert_type;
    subplot(2,npert_types,isubplot)
    hist(xcspec(ipert_type).tonset,20)
    axis([min_t max_t 0 max_bincounts]);
    if ipert_type == 1, ylabel('tonset'); end
    title(sprintf('ipert(%d)',ipert_type));
    isubplot = 1*npert_types + ipert_type;
    subplot(2,npert_types,isubplot)
    hist(xcspec(ipert_type).tpeak,20)
    axis([min_t max_t 0 max_bincounts]);
    if ipert_type == 1, ylabel('tpeak'); end
  end
  
  figure
  npert_types = pert_resp.npert_types;
  for ipert_type = 1:npert_types
    ntrials = pert_resp.n_good_trials(ipert_type);
    y = xcspec(ipert_type).peakcomp';
    x = pert_resp.comp{ipert_type};
    X = [ones(ntrials,1) x];
    [b,duh,duh2,duh3,stats] = regress(y,X);
    r = sign(mean(b(2)))*sqrt(stats(1));
    p = stats(3);
    subplot(1,npert_types,ipert_type)
    plot(x,y,'*')
    yfit(1) = b(2)*min(x) + b(1);
    yfit(2) = b(2)*max(x) + b(1);
    hl = line([min(x) max(x)],yfit);
    set(hl,'Color','r');
    ht = title(sprintf('ipert(%d): m(%.3f),r(%.3f),p(%f)',ipert_type,b(2),r,p)); set(ht,'Interpreter','none');
    ht = xlabel('comp'); set(ht,'Interpreter','none');
    ht = ylabel('peakcomp'); set(ht,'Interpreter','none');
  end
  
  figure
  npert_types = pert_resp.npert_types;
  for ipert_type = 1:npert_types
    ntrials = pert_resp.n_good_trials(ipert_type);
    y = xcspec(ipert_type).peakresp';
    x = pert_resp.mean_signed_response{ipert_type};
    X = [ones(ntrials,1) x];
    [b,duh,duh2,duh3,stats] = regress(y,X);
    r = sign(mean(b(2)))*sqrt(stats(1));
    p = stats(3);
    subplot(1,npert_types,ipert_type)
    plot(x,y,'*')
    yfit(1) = b(2)*min(x) + b(1);
    yfit(2) = b(2)*max(x) + b(1);
    hl = line([min(x) max(x)],yfit);
    set(hl,'Color','r');
    ht = title(sprintf('ipert(%d): m(%.3f),r(%.3f),p(%f)',ipert_type,b(2),r,p)); set(ht,'Interpreter','none');
    ht = xlabel('mean_signed_response'); set(ht,'Interpreter','none');
    ht = ylabel('peakresp'); set(ht,'Interpreter','none');
  end
end

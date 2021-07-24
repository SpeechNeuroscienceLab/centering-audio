function make_baseline4comp(pert_resp,colors,fig_pos)

npert_types = pert_resp.npert_types;
parsed_frame_taxis = pert_resp.frame_taxis;

reply = input('baseline type? [(l)inear]/(m)ean response : ','s');
if isempty(reply) || strcmp(reply(1),'l')
  baseline_type = 'l';
else
  baseline_type = 'm';
end

mean_parsed_pitch_in =  pert_resp.cents.pitch_in.mean((npert_types+1),:);
stde_parsed_pitch_in =  pert_resp.cents.pitch_in.stde((npert_types+1),:);
mean_parsed_pitch_out = pert_resp.cents.pitch_out.mean((npert_types+1),:);
stde_parsed_pitch_out = pert_resp.cents.pitch_out.stde((npert_types+1),:);

meanresp_baseline = mean_parsed_pitch_in;

if strcmp(baseline_type,'m')
  tlims4baseline = [];
  baseline = meanresp_baseline;
  linear_baseline = [];
else
  tlims4baseline = [parsed_frame_taxis(1) 0];
  hf = figure;
  set(hf,'Position',fig_pos);
  while 1
    ilims4baseline = dsearchn(parsed_frame_taxis',tlims4baseline')';
    idxes4baseline = ilims4baseline(1):ilims4baseline(2);
    fitcoffs4basline = polyfit(parsed_frame_taxis(idxes4baseline),mean_parsed_pitch_in(idxes4baseline),1);
    linear_baseline = polyval(fitcoffs4basline,parsed_frame_taxis);
    clf
    
    hax1 = subplot(211);
    for ipert_type = 1:npert_types
      hpl = plot(parsed_frame_taxis,pert_resp.cents.pitch_out.mean(ipert_type,:),colors{ipert_type}); set(hpl,'LineWidth',3);
      hold on
      hpl = plot(parsed_frame_taxis,pert_resp.cents.pitch_out.mean(ipert_type,:) + pert_resp.cents.pitch_out.stde(ipert_type,:),colors{ipert_type}); set(hpl,'LineWidth',1);
      hpl = plot(parsed_frame_taxis,pert_resp.cents.pitch_out.mean(ipert_type,:) - pert_resp.cents.pitch_out.stde(ipert_type,:),colors{ipert_type}); set(hpl,'LineWidth',1);
    end
    hpl = plot(parsed_frame_taxis,linear_baseline,'g'); set(hpl,'LineWidth',3);
    hl = vline(tlims4baseline(1),'g'); hl = vline(tlims4baseline(2),'g');
    
    hax2 = subplot(212);
    hpl = plot(parsed_frame_taxis,mean_parsed_pitch_in,'k'); set(hpl,'LineWidth',3);
    hold on
    hpl = plot(parsed_frame_taxis,mean_parsed_pitch_in + stde_parsed_pitch_in,'k'); set(hpl,'LineWidth',1);
    hpl = plot(parsed_frame_taxis,mean_parsed_pitch_in - stde_parsed_pitch_in,'k'); set(hpl,'LineWidth',1);
    for ipert_type = 1:npert_types
      hpl = plot(parsed_frame_taxis,pert_resp.cents.pitch_in.mean(ipert_type,:),colors{ipert_type}); set(hpl,'LineWidth',3);
      hpl = plot(parsed_frame_taxis,pert_resp.cents.pitch_in.mean(ipert_type,:) + pert_resp.cents.pitch_in.stde(ipert_type,:),colors{ipert_type}); set(hpl,'LineWidth',1);
      hpl = plot(parsed_frame_taxis,pert_resp.cents.pitch_in.mean(ipert_type,:) - pert_resp.cents.pitch_in.stde(ipert_type,:),colors{ipert_type}); set(hpl,'LineWidth',1);
    end
    hpl = plot(parsed_frame_taxis,linear_baseline,'g'); set(hpl,'LineWidth',3);
    hl = vline(tlims4baseline(1),'g'); hl = vline(tlims4baseline(2),'g');
    
    reply = input('good baseline? [y]/n: ','s');
    if isempty(reply) || strcmp(reply,'y'), break; end
    fprintf('set lower and upper time limits for baseline\n');
    [xx,yy] = ginput(2);
    tlims4baseline = xx';
  end
  baseline = linear_baseline;
end
save('baseline4comp','baseline','tlims4baseline','linear_baseline','meanresp_baseline','baseline_type');

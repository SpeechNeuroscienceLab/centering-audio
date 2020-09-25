function make_tlims4comp2(pert_resp,colors,fig_pos)

parsed_frame_taxis = pert_resp.frame_taxis;
npert_types = pert_resp.npert_types;
ncolors = length(colors);

hf = figure;
set(hf,'Position',fig_pos);
tlims4comp = [0.4 0.8];
while 1
  clf
  hax1 = subplot(211);
  for ipert_type = 1:npert_types
    color_idx = rem(ipert_type,ncolors); if ~color_idx, color_idx = ncolors; end
    n_pert_resp = pert_resp.n_good_trials(ipert_type);
    for i_pert_resp = 1:n_pert_resp
      plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_out.dat{ipert_type}(i_pert_resp,:),colors{color_idx});
      hold on
    end
    hpl = plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_out.mean(ipert_type,:),'k'); set(hpl,'LineWidth',3);
    hpl = plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_out.mean(ipert_type,:) + pert_resp.cents4comp.pitch_out.stde(ipert_type,:),'k'); set(hpl,'LineWidth',1);
    hpl = plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_out.mean(ipert_type,:) - pert_resp.cents4comp.pitch_out.stde(ipert_type,:),'k'); set(hpl,'LineWidth',1);
  end
  hl = vline(tlims4comp(1),'g'); hl = vline(tlims4comp(2),'g');
  hax2 = subplot(212);
  for ipert_type = 1:npert_types
    color_idx = rem(ipert_type,ncolors); if ~color_idx, color_idx = ncolors; end
    n_pert_resp = pert_resp.n_good_trials(ipert_type);
    for i_pert_resp = 1:n_pert_resp
      plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.dat{ipert_type}(i_pert_resp,:),colors{color_idx});
      hold on
    end
    hpl = plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.mean(ipert_type,:),'k'); set(hpl,'LineWidth',3);
    hpl = plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.mean(ipert_type,:) + pert_resp.cents4comp.pitch_in.stde(ipert_type,:),'k'); set(hpl,'LineWidth',1);
    hpl = plot(parsed_frame_taxis,pert_resp.cents4comp.pitch_in.mean(ipert_type,:) - pert_resp.cents4comp.pitch_in.stde(ipert_type,:),'k'); set(hpl,'LineWidth',1);
  end
  hl = vline(tlims4comp(1),'g'); hl = vline(tlims4comp(2),'g');

  reply = input('good time window for measuring comp? [y]/n: ','s');
  if isempty(reply) || strcmp(reply,'y'), break; end
  fprintf('set lower and upper time limits for window\n');
  [xx,yy] = ginput(2);
  tlims4comp = xx';
end
ilims4comp = dsearchn(parsed_frame_taxis',tlims4comp')';
idxes4comp = ilims4comp(1):ilims4comp(2);
save('tlims4comp','tlims4comp','ilims4comp','idxes4comp');

if ~exist('centsdev_dat','var')
  load('centsdev_dat');
end

frame_taxis = centsdev_dat.frame_taxis;

datrangelims_nstdvs = 2;
yes_cull_patients = 0; % no patient cull results from setting this to 1 with datrangelims_nstdvs = 2
yes_cull_controls = 0;
yes_2ndcull_controls = 0;

alphalev = 0.05;

% tlims4anal_req = [0.1 0.9];
% tlims4anal_req = [0.1 2.0];
%tlims4anal_req = [0.05 1.0];
tlims4anal_req = [-0.2 1.0];
ilims4anal = dsearchn(frame_taxis',tlims4anal_req')';
% idxes4anal = ilims4anal(1):ilims4anal(2);
idxes4anal = ilims4anal(1):(ilims4anal(2) - 2); % hack because I just discovered Inf's in the last two cols of patient absperttrial data
tlims4anal = frame_taxis(ilims4anal);

% tinc4segs = 0.0125/2;
% tinc4segs = 0.0125;
% tinc4segs = 0.025;
tinc4segs = 0.05;
% tinc4segs = 0.1;
tlims4segs_req = [(tlims4anal(1):tinc4segs:(tlims4anal(2) - tinc4segs))' ...
                  ((tlims4anal(1) + tinc4segs):tinc4segs:tlims4anal(2))'];
[nsegs,duh] = size(tlims4segs_req);
for iseg = 1:nsegs
  ilims4segs(iseg,:) = dsearchn(frame_taxis',tlims4segs_req(iseg,:)')';
  idxes4segs{iseg}   = ilims4segs(iseg,1):ilims4segs(iseg,2);
  tlims4segs(iseg,:) = frame_taxis(ilims4segs(iseg,:));
end

good_patient_idxes = find(centsdev_dat.patients.is_good); ngood_patients = length(good_patient_idxes);
good_control_idxes = find(centsdev_dat.controls.is_good); ngood_controls = length(good_control_idxes);

nframes = centsdev_dat.nframeswin;

ipert_type = [1 2];
% ipert_type = [1];
% ipert_type = [2];
nperts = length(ipert_type);

% gather the patient data
clear patient_perttrial
for ipert = 1:nperts
  itrial_last = 0;
  for isubj = 1:ngood_patients
    the_subj = good_patient_idxes(isubj);
    the_pert = ipert_type(ipert);
    the_centsdev_dat  = centsdev_dat.patients.subj(the_subj).dat{the_pert};
    ngood_trials4subj = centsdev_dat.patients.subj(the_subj).n_good_trials(the_pert);
    perttrial_idxes4subj = (itrial_last+1):(itrial_last+ngood_trials4subj);
    patient_perttrial{ipert}.dat(perttrial_idxes4subj,:) = the_centsdev_dat;
    if ipert == 1
      patient_perttrial{ipert}.absdat(perttrial_idxes4subj,:) = -the_centsdev_dat;
    else
      patient_perttrial{ipert}.absdat(perttrial_idxes4subj,:) = +the_centsdev_dat;
    end
    patient_perttrial{ipert}.subj(perttrial_idxes4subj,1) = the_subj*ones([ngood_trials4subj 1]);
    itrial_last = itrial_last + ngood_trials4subj;
  end
  patient_perttrial{ipert}.ntrials = itrial_last;
end

% display, and optionally cull the patient data (I'd prefer not to do this)
nhf = 0;
nhf = nhf + 1; hf(nhf) = figure;
nhf_rawdat_patients = nhf;
for ipert = 1:nperts
  patient_perttrial{ipert}.premean = nanmean(patient_perttrial{ipert}.dat,  1);
  for itrial = 1:patient_perttrial{ipert}.ntrials
    patient_perttrial{ipert}.dev_from_premean(itrial,:) = ...
        patient_perttrial{ipert}.dat(itrial,:) - patient_perttrial{ipert}.premean;
  end
  patient_perttrial{ipert}.premeanstdv = mean(nanstd( patient_perttrial{ipert}.dat,0,1));
  
  hax(nhf,ipert) = subplot(2,1,ipert);
  plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.dev_from_premean(:,idxes4anal)');
  kids = get(gca,'Children');
  if yes_cull_patients
    datrangelims = datrangelims_nstdvs*patient_perttrial{ipert}.premeanstdv*[-1 1];
    hl = hline(datrangelims(1),'k');
    hl = hline(datrangelims(2),'k');
    in_range_dat = (datrangelims(1) < patient_perttrial{ipert}.dev_from_premean(:,idxes4anal)) & ...
                   (patient_perttrial{ipert}.dev_from_premean(:,idxes4anal) < datrangelims(2));
    out_of_range_dat = ~in_range_dat;
    out_of_range_trials = any(out_of_range_dat')';
    set(kids(out_of_range_trials(end:(-1):1)),'Color',0.7*[1 1 1]);
    patient_perttrial{ipert}.dat(out_of_range_dat) = NaN;
    patient_perttrial{ipert}.absdat(out_of_range_dat) = NaN;
  end
  axis([tlims4anal -400 400]);
  patient_perttrial{ipert}.ntrials = patient_perttrial{ipert}.ntrials - sum(isnan(patient_perttrial{ipert}.dat(:,1)));
end

% consolidate the patient absolute pert response data
patient_absperttrial.dat =    [patient_perttrial{1}.absdat;   patient_perttrial{2}.absdat];
patient_absperttrial.subj =   [patient_perttrial{1}.subj;     patient_perttrial{2}.subj];
patient_absperttrial.ntrials = patient_perttrial{1}.ntrials + patient_perttrial{2}.ntrials;

% gather the control data
clear control_perttrial
for ipert = 1:nperts
  itrial_last = 0;
  for isubj = 1:ngood_controls
    the_subj = good_control_idxes(isubj);
    the_pert = ipert_type(ipert);
    the_centsdev_dat = centsdev_dat.controls.subj(the_subj).dat{the_pert};
    ngood_trials4subj = centsdev_dat.controls.subj(the_subj).n_good_trials(the_pert);
    perttrial_idxes4subj = (itrial_last+1):(itrial_last+ngood_trials4subj);
    control_perttrial{ipert}.dat(perttrial_idxes4subj,:) = the_centsdev_dat;
    if ipert == 1
      control_perttrial{ipert}.absdat(perttrial_idxes4subj,:) = -the_centsdev_dat;
    else
      control_perttrial{ipert}.absdat(perttrial_idxes4subj,:) = +the_centsdev_dat;
    end
    % use (the_subj + ngood_patients) here to make control subject IDs different from patient subject IDs
    % control_perttrial{ipert}.subj(perttrial_idxes4subj,:) = (the_subj + ngood_patients)*ones([ngood_trials4subj 1]);
    control_perttrial{ipert}.subj(perttrial_idxes4subj,:) = (the_subj)*ones([ngood_trials4subj 1]);
    itrial_last = itrial_last + ngood_trials4subj;
  end
  control_perttrial{ipert}.ntrials = itrial_last;
end

% display and optionally cull the control data (I'd prefer not to do this.)
nhf = nhf + 1; hf(nhf) = figure;
nhf_rawdat_controls = nhf;
for ipert = 1:nperts
  control_perttrial{ipert}.premean = nanmean(control_perttrial{ipert}.dat,  1);
  for itrial = 1:control_perttrial{ipert}.ntrials
    control_perttrial{ipert}.dev_from_premean(itrial,:) = ...
        control_perttrial{ipert}.dat(itrial,:) - control_perttrial{ipert}.premean;
  end
  control_perttrial{ipert}.premeanstdv = mean(nanstd( control_perttrial{ipert}.dat,0,1));
  
  hax(nhf,ipert) = subplot(2,1,ipert);
  plot(frame_taxis(idxes4anal),control_perttrial{ipert}.dev_from_premean(:,idxes4anal)');
  kids = get(gca,'Children');
  if yes_cull_controls
    datrangelims = datrangelims_nstdvs*control_perttrial{ipert}.premeanstdv*[-1 1];
    hl = hline(datrangelims(1),'k');
    hl = hline(datrangelims(2),'k');
    in_range_dat = (datrangelims(1) < control_perttrial{ipert}.dev_from_premean(:,idxes4anal)) & ...
                   (control_perttrial{ipert}.dev_from_premean(:,idxes4anal) < datrangelims(2));
    out_of_range_dat = ~in_range_dat;
    out_of_range_trials = any(out_of_range_dat')';
    set(kids(out_of_range_trials(end:(-1):1)),'Color',0.7*[1 1 1]);
    control_perttrial{ipert}.dat(out_of_range_dat) = NaN;
    control_perttrial{ipert}.absdat(out_of_range_dat) = NaN;
  end
  axis([tlims4anal -400 400]);
  control_perttrial{ipert}.ntrials = control_perttrial{ipert}.ntrials - sum(isnan(control_perttrial{ipert}.dat(:,1)));
end

% optionally to a "second cull"(??) of the control data (I'm not doing this.)
if yes_cull_controls && yes_2ndcull_controls
  nhf = nhf + 1; hf(nhf) = figure;
  nhf_2ndcull_controls = nhf;
  for ipert = 1:nperts
    hax(nhf,ipert) = subplot(2,1,ipert);
    plot(frame_taxis(idxes4anal),control_perttrial{ipert}.dat(:,idxes4anal)');
    kids = get(gca,'Children');
    hl = hline(datrangelims(1),'k');
    hl = hline(100,'k');
    in_range_trials = (datrangelims(1) < control_perttrial{ipert}.dat(:,idxes4anal(end))) & ...
                      (control_perttrial{ipert}.dat(:,idxes4anal(end)) < 100);
    out_of_range_trials = ~in_range_trials;
    set(kids(out_of_range_trials(end:(-1):1)),'Color',0.7*[1 1 1]);
    axis([tlims4anal -400 400]);
    
    control_perttrial{ipert}.dat(out_of_range_trials,:) = [];
    control_perttrial{ipert}.absdat(out_of_range_trials,:) = [];
    control_perttrial{ipert}.subj(out_of_range_trials) = [];
    control_perttrial{ipert}.ntrials = length(control_perttrial{ipert}.subj);
  end
end

% consolidate the control absolute pert response data
control_absperttrial.dat =     [control_perttrial{1}.absdat;   control_perttrial{2}.absdat];
control_absperttrial.subj =    [control_perttrial{1}.subj;     control_perttrial{2}.subj];
control_absperttrial.ntrials =  control_perttrial{1}.ntrials + control_perttrial{2}.ntrials;

for ipert = 1:nperts
  
  patient_perttrial{ipert}.mean = nanmean(patient_perttrial{ipert}.dat,1);
  patient_perttrial{ipert}.stde = nanstd(patient_perttrial{ipert}.dat,0,1)/sqrt(patient_perttrial{ipert}.ntrials);
  control_perttrial{ipert}.mean = nanmean(control_perttrial{ipert}.dat,1);
  control_perttrial{ipert}.stde = nanstd(control_perttrial{ipert}.dat,0,1)/sqrt(control_perttrial{ipert}.ntrials);

  patient_perttrial{ipert}.absmean = nanmean(patient_perttrial{ipert}.absdat,1);
  patient_perttrial{ipert}.absstde = nanstd(patient_perttrial{ipert}.absdat,0,1)/sqrt(patient_perttrial{ipert}.ntrials);
  control_perttrial{ipert}.absmean = nanmean(control_perttrial{ipert}.absdat,1);
  control_perttrial{ipert}.absstde = nanstd(control_perttrial{ipert}.absdat,0,1)/sqrt(control_perttrial{ipert}.ntrials);

  all_perttrial{ipert}.mean = nanmean([patient_perttrial{ipert}.mean; control_perttrial{ipert}.mean],1);
end

patient_absperttrial.mean = nanmean(patient_absperttrial.dat,1);
patient_absperttrial.stde = nanstd( patient_absperttrial.dat,0,1)/sqrt(patient_absperttrial.ntrials);
control_absperttrial.mean = nanmean(control_absperttrial.dat,1);
control_absperttrial.stde = nanstd( control_absperttrial.dat,0,1)/sqrt(control_absperttrial.ntrials);
    all_absperttrial.mean = nanmean([patient_absperttrial.mean; control_absperttrial.mean],1);

% calculate deviations from all_perttrial{ipert}.mean and all_absperttrial.mean
for ipert = 1:nperts
  for itrial = 1:patient_perttrial{ipert}.ntrials
    patient_perttrial{ipert}.dev(itrial,:) = patient_perttrial{ipert}.dat(itrial,:) - all_perttrial{ipert}.mean;
  end
  for itrial = 1:control_perttrial{ipert}.ntrials
    control_perttrial{ipert}.dev(itrial,:) = control_perttrial{ipert}.dat(itrial,:) - all_perttrial{ipert}.mean;
  end
end
for itrial = 1:patient_absperttrial.ntrials
  patient_absperttrial.dev(itrial,:) = patient_absperttrial.dat(itrial,:) - all_absperttrial.mean;
end
for itrial = 1:control_absperttrial.ntrials
  control_absperttrial.dev(itrial,:) = control_absperttrial.dat(itrial,:) - all_absperttrial.mean;
end
for ipert = 1:nperts
  patient_perttrial{ipert}.devmean = nanmean(patient_perttrial{ipert}.dev,1);
  patient_perttrial{ipert}.devstde = nanstd(patient_perttrial{ipert}.dev,0,1)/sqrt(patient_perttrial{ipert}.ntrials);
  control_perttrial{ipert}.devmean = nanmean(control_perttrial{ipert}.dev,1);
  control_perttrial{ipert}.devstde = nanstd(control_perttrial{ipert}.dev,0,1)/sqrt(control_perttrial{ipert}.ntrials);
end
patient_absperttrial.devmean = nanmean(patient_absperttrial.dev,1);
patient_absperttrial.devstde = nanstd( patient_absperttrial.dev,0,1)/sqrt(patient_absperttrial.ntrials);
control_absperttrial.devmean = nanmean(control_absperttrial.dev,1);
control_absperttrial.devstde = nanstd( control_absperttrial.dev,0,1)/sqrt(control_absperttrial.ntrials);

nhf = nhf + 1; hf(nhf) = figure;
nhf_meandat = nhf;
set(nhf_meandat,'Name','diff btw groups for each pert response');
for ipert = 1:nperts
  hax(nhf,ipert) = subplot(2,1,ipert);
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.mean(idxes4anal),'b');
  hold on
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.mean(idxes4anal),'r');

  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.mean(idxes4anal) + patient_perttrial{ipert}.stde(idxes4anal),'b');
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.mean(idxes4anal) - patient_perttrial{ipert}.stde(idxes4anal),'b');
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.mean(idxes4anal) + control_perttrial{ipert}.stde(idxes4anal),'r');
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.mean(idxes4anal) - control_perttrial{ipert}.stde(idxes4anal),'r');
  a = axis; axis([tlims4anal a(3:4)]);
end

nhf = nhf + 1; hf(nhf) = figure;
nhf_meandat2 = nhf;
set(nhf_meandat2,'Name','diff btw pert reseponses for each group');
hax(nhf,1) = subplot(2,1,1);
for ipert = 1:nperts
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.mean(idxes4anal),'b');
  hold on
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.mean(idxes4anal) + patient_perttrial{ipert}.stde(idxes4anal),'b');
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.mean(idxes4anal) - patient_perttrial{ipert}.stde(idxes4anal),'b');
end
a = axis; axis([tlims4anal a(3:4)]);
hax(nhf,2) = subplot(2,1,2);
for ipert = 1:nperts
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.mean(idxes4anal),'r');
  hold on
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.mean(idxes4anal) + control_perttrial{ipert}.stde(idxes4anal),'r');
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.mean(idxes4anal) - control_perttrial{ipert}.stde(idxes4anal),'r');
end
a = axis; axis([tlims4anal a(3:4)]);

nhf = nhf + 1; hf(nhf) = figure;
nhf_absmeandat2 = nhf;
set(nhf_absmeandat2,'Name','diff btw abs pert responses for each group');
hax(nhf,1) = subplot(2,1,1);
for ipert = 1:nperts
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.absmean(idxes4anal),'b');
  hold on
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.absmean(idxes4anal) + patient_perttrial{ipert}.absstde(idxes4anal),'b');
  hpl = plot(frame_taxis(idxes4anal),patient_perttrial{ipert}.absmean(idxes4anal) - patient_perttrial{ipert}.absstde(idxes4anal),'b');
end
a = axis; axis([tlims4anal a(3:4)]);
hax(nhf,2) = subplot(2,1,2);
for ipert = 1:nperts
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.absmean(idxes4anal),'r');
  hold on
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.absmean(idxes4anal) + control_perttrial{ipert}.absstde(idxes4anal),'r');
  hpl = plot(frame_taxis(idxes4anal),control_perttrial{ipert}.absmean(idxes4anal) - control_perttrial{ipert}.absstde(idxes4anal),'r');
end
a = axis; axis([tlims4anal a(3:4)]);

nhf = nhf + 1; hf(nhf) = figure;
nhf_absmeandat3 = nhf;
set(nhf_absmeandat3,'Name','diff btw groups for consolidated abs pert responses');
hax(nhf,1) = axes;
hpl = plot(frame_taxis(idxes4anal),patient_absperttrial.mean(idxes4anal),'b');
hold on
hpl = plot(frame_taxis(idxes4anal),patient_absperttrial.mean(idxes4anal) + patient_absperttrial.stde(idxes4anal),'b');
hpl = plot(frame_taxis(idxes4anal),patient_absperttrial.mean(idxes4anal) - patient_absperttrial.stde(idxes4anal),'b');
hpl = plot(frame_taxis(idxes4anal),control_absperttrial.mean(idxes4anal),'r');
hpl = plot(frame_taxis(idxes4anal),control_absperttrial.mean(idxes4anal) + control_absperttrial.stde(idxes4anal),'r');
hpl = plot(frame_taxis(idxes4anal),control_absperttrial.mean(idxes4anal) - control_absperttrial.stde(idxes4anal),'r');

hpl = plot(frame_taxis(idxes4anal),all_absperttrial.mean(idxes4anal),'k'); set(hpl,'LineWidth',3);
a = axis; axis([tlims4anal a(3:4)]);

yes_plot_devs = 1;
if yes_plot_devs
  nhf = nhf + 1; hf(nhf) = figure;
  nhf_devmeandat = nhf;

  hax(nhf,1) = axes;
  hpl = plot(frame_taxis(idxes4anal),patient_absperttrial.devmean(idxes4anal),'b');
  hold on
  hpl = plot(frame_taxis(idxes4anal),patient_absperttrial.devmean(idxes4anal) + patient_absperttrial.devstde(idxes4anal),'b');
  hpl = plot(frame_taxis(idxes4anal),patient_absperttrial.devmean(idxes4anal) - patient_absperttrial.devstde(idxes4anal),'b');
  hpl = plot(frame_taxis(idxes4anal),control_absperttrial.devmean(idxes4anal),'r');
  hpl = plot(frame_taxis(idxes4anal),control_absperttrial.devmean(idxes4anal) + control_absperttrial.devstde(idxes4anal),'r');
  hpl = plot(frame_taxis(idxes4anal),control_absperttrial.devmean(idxes4anal) - control_absperttrial.devstde(idxes4anal),'r');
  a = axis; axis([tlims4anal a(3:4)]);
end

for iseg = 1:nsegs
  for ipert = 1:nperts
    for itrial = 1:patient_perttrial{ipert}.ntrials
      patient_perttrial{ipert}.segmean(itrial,iseg) = nanmean(patient_perttrial{ipert}.dat(itrial,idxes4segs{iseg}),2);
      patient_perttrial{ipert}.segabsmean(itrial,iseg) = nanmean(patient_perttrial{ipert}.absdat(itrial,idxes4segs{iseg}),2);
      patient_perttrial{ipert}.segdevmean(itrial,iseg) = nanmean(patient_perttrial{ipert}.dev(itrial,idxes4segs{iseg}),2);
    end
    for itrial = 1:control_perttrial{ipert}.ntrials
      control_perttrial{ipert}.segmean(itrial,iseg) = nanmean(control_perttrial{ipert}.dat(itrial,idxes4segs{iseg}),2);
      control_perttrial{ipert}.segabsmean(itrial,iseg) = nanmean(control_perttrial{ipert}.absdat(itrial,idxes4segs{iseg}),2);
      control_perttrial{ipert}.segdevmean(itrial,iseg) = nanmean(control_perttrial{ipert}.dev(itrial,idxes4segs{iseg}),2);
    end
  end
  for itrial = 1:patient_absperttrial.ntrials
    patient_absperttrial.segmean(itrial,iseg) = nanmean(patient_absperttrial.dat(itrial,idxes4segs{iseg}),2);
    patient_absperttrial.segdevmean(itrial,iseg) = nanmean(patient_absperttrial.dev(itrial,idxes4segs{iseg}),2);
  end
  for itrial = 1:control_absperttrial.ntrials
    control_absperttrial.segmean(itrial,iseg) = nanmean(control_absperttrial.dat(itrial,idxes4segs{iseg}),2);
    control_absperttrial.segdevmean(itrial,iseg) = nanmean(control_absperttrial.dev(itrial,idxes4segs{iseg}),2);
  end
end

for ipert = 1:nperts
  for iseg = 1:nsegs
    groupseg_anova_idat = 0;
    for itrial = 1:patient_perttrial{ipert}.ntrials
      groupseg_anova_idat = groupseg_anova_idat + 1;
      groupseg_anova_datY(groupseg_anova_idat) = patient_perttrial{ipert}.segmean(itrial,iseg);
      ng = 0;
      ng = ng + 1; groupseg_anova_G{ng}(groupseg_anova_idat) = 'p';
    end
    for itrial = 1:control_perttrial{ipert}.ntrials
      groupseg_anova_idat = groupseg_anova_idat + 1;
      groupseg_anova_datY(groupseg_anova_idat) = control_perttrial{ipert}.segmean(itrial,iseg);
      ng = 0;
      ng = ng + 1; groupseg_anova_G{ng}(groupseg_anova_idat) = 'c';
    end
    groupseg_varnames = {'group'};
    dims2comp = [1];
    [groupseg_p(ipert,iseg),groupseg_tab{ipert,iseg},groupseg_stats{ipert,iseg},groupseg_terms{ipert,iseg}] = ...
        anovan(groupseg_anova_datY,groupseg_anova_G,'varnames',groupseg_varnames,'display','off');
  end
end

for iseg = 1:nsegs
  for ipert = 1:nperts
    if groupseg_p(ipert,iseg) < alphalev
      axes(hax(nhf_meandat,ipert));
      hp = vpatch(tlims4segs(iseg,1),tlims4segs(iseg,2),[.7 .9 .7]);
      move2back([],hp);
    end
  end
end

for igroup = 1:2
  clear pertseg_anova_datY pertsegabs_anova_datY pertseg_anova_devY pertseg_anova_G
  for iseg = 1:nsegs
    pertseg_anova_idat = 0;
    for ipert = 1:nperts
      if igroup == 1
        for itrial = 1:patient_perttrial{ipert}.ntrials
          pertseg_anova_idat = pertseg_anova_idat + 1;
          pertseg_anova_datY(pertseg_anova_idat) = patient_perttrial{ipert}.segmean(itrial,iseg);
          pertsegabs_anova_datY(pertseg_anova_idat) = patient_perttrial{ipert}.segabsmean(itrial,iseg);
          ng = 0;
          ng = ng + 1; pertseg_anova_G{ng}(pertseg_anova_idat) = ipert;
        end
      else
        for itrial = 1:control_perttrial{ipert}.ntrials
          pertseg_anova_idat = pertseg_anova_idat + 1;
          pertseg_anova_datY(pertseg_anova_idat) = control_perttrial{ipert}.segmean(itrial,iseg);
          pertsegabs_anova_datY(pertseg_anova_idat) = control_perttrial{ipert}.segabsmean(itrial,iseg);
          ng = 0;
          ng = ng + 1; pertseg_anova_G{ng}(pertseg_anova_idat) = ipert;
        end
      end
    end
    pertseg_varnames = {'pert'};
    dims2comp = [1];
    [pertseg_p(igroup,iseg),pertseg_tab{igroup,iseg},pertseg_stats{igroup,iseg},pertseg_terms{igroup,iseg}] = ...
        anovan(pertseg_anova_datY,pertseg_anova_G,'varnames',pertseg_varnames,'display','off');
    [pertsegabs_p(igroup,iseg),pertsegabs_tab{igroup,iseg},pertsegabs_stats{igroup,iseg},pertsegabs_terms{igroup,iseg}] = ...
        anovan(pertsegabs_anova_datY,pertseg_anova_G,'varnames',pertseg_varnames,'display','off');
  end
end

for igroup = 1:2
  for iseg = 1:nsegs
    if pertseg_p(igroup,iseg) < alphalev
      axes(hax(nhf_meandat2,igroup));
      hp = vpatch(tlims4segs(iseg,1),tlims4segs(iseg,2),[.7 .9 .7]);
      move2back([],hp);
    end
    if pertsegabs_p(igroup,iseg) < 0.05 % separate alphalev here
      axes(hax(nhf_absmeandat2,igroup));
      hp = vpatch(tlims4segs(iseg,1),tlims4segs(iseg,2),[.7 .9 .7]);
      move2back([],hp);
    end
  end
end

% adding 'subject' (subj ID) as a random factor doesn't work (gives NaN p values)
% I think because the differing subject IDs "sucks up" all the available variance
% so leave the 'subject' related lines below commented out.
% maybe try non-parametric methods
clear absgroupseg_anova_datY absgroupseg_anova_devY absgroupseg_anova_G absgroupseg_p
for iseg = 1:nsegs
  absgroupseg_anova_idat = 0;
  for itrial = 1:patient_absperttrial.ntrials
    absgroupseg_anova_idat = absgroupseg_anova_idat + 1;
    absgroupseg_anova_datY(absgroupseg_anova_idat) = patient_absperttrial.segmean(itrial,iseg);
    absgroupseg_anova_devY(absgroupseg_anova_idat) = patient_absperttrial.segdevmean(itrial,iseg);
    ng = 0;
    ng = ng + 1; absgroupseg_anova_G{ng}(absgroupseg_anova_idat) = 'p';
    % ng = ng + 1; absgroupseg_anova_G{ng}(absgroupseg_anova_idat) = patient_absperttrial.subj(itrial);
  end
  for itrial = 1:control_absperttrial.ntrials
    absgroupseg_anova_idat = absgroupseg_anova_idat + 1;
    absgroupseg_anova_datY(absgroupseg_anova_idat) = control_absperttrial.segmean(itrial,iseg);
    absgroupseg_anova_devY(absgroupseg_anova_idat) = control_absperttrial.segdevmean(itrial,iseg);
    ng = 0;
    ng = ng + 1; absgroupseg_anova_G{ng}(absgroupseg_anova_idat) = 'c';
    % ng = ng + 1; absgroupseg_anova_G{ng}(absgroupseg_anova_idat) = control_absperttrial.subj(itrial);
  end
  absgroupseg_varnames = {'group'};
  % absgroupseg_varnames = {'group' 'subject'};
  dims2comp = [1];
  [absgroupseg_p{iseg},absgroupseg_tab{iseg},absgroupseg_stats{iseg},absgroupseg_terms{iseg}] = ...
      anovan(absgroupseg_anova_datY,absgroupseg_anova_G,'varnames',absgroupseg_varnames,'display','off'); % 'model',2,'random',2);
  [devabsgroupseg_p{iseg},devabsgroupseg_tab{iseg},devabsgroupseg_stats{iseg},devabsgroupseg_terms{iseg}] = ...
      anovan(absgroupseg_anova_devY,absgroupseg_anova_G,'varnames',absgroupseg_varnames,'display','off'); % 'model',2,'random',2);
end

for iseg = 1:nsegs
  if absgroupseg_p{iseg}(1) < alphalev
    axes(hax(nhf_absmeandat3,1));
    hp = vpatch(tlims4segs(iseg,1),tlims4segs(iseg,2),[.7 .9 .7]);
    move2back([],hp);
  end
end

if yes_plot_devs
  in_sig_itvl = 0;
  n_sig_itvls = 0;
  for iseg = 1:nsegs
    if devabsgroupseg_p{iseg}(1) < alphalev
      axes(hax(nhf_devmeandat,1));
      hp = vpatch(tlims4segs(iseg,1),tlims4segs(iseg,2),[.7 .9 .7]);
      move2back([],hp);
      if ~in_sig_itvl
        in_sig_itvl = 1;
        n_sig_itvls = n_sig_itvls + 1;
        sig_itvl(n_sig_itvls,1) = iseg;
        sig_itvl(n_sig_itvls,2) = iseg;
      else
        sig_itvl(n_sig_itvls,2) = iseg;
      end
    else
      in_sig_itvl = 0;
    end
  end
  n_stat_itvls = 0;
  for i_sig_itvl = 1:n_sig_itvls
    if tlims4segs(sig_itvl(i_sig_itvl,1),1) > 0.050 % milliseconds
      if sig_itvl(i_sig_itvl,2) - sig_itvl(i_sig_itvl,1) > 1
        n_stat_itvls = n_stat_itvls + 1;
        stat_itvl(n_stat_itvls,1) = sig_itvl(i_sig_itvl,1) + 1;
        stat_itvl(n_stat_itvls,2) = sig_itvl(i_sig_itvl,2) - 1;
      end
    end
  end
  stat_itvl_anova_idat = 0;
  for i_stat_itvl = 1:n_stat_itvls
    iseg_lo = stat_itvl(i_stat_itvl,1);
    iseg_hi = stat_itvl(i_stat_itvl,2);
    hl = vline(tlims4segs(iseg_lo,1),'k');
    hl = vline(tlims4segs(iseg_hi,2),'k');
    for itrial = 1:patient_absperttrial.ntrials
      stat_itvl_anova_idat = stat_itvl_anova_idat + 1;
      stat_itvl_anova_devY(stat_itvl_anova_idat) = mean(patient_absperttrial.segdevmean(itrial,iseg_lo:iseg_hi),2);
      ng = 0;
      ng = ng + 1; stat_itvl_anova_G{ng}(stat_itvl_anova_idat) = 'p';
      ng = ng + 1; stat_itvl_anova_G{ng}(stat_itvl_anova_idat) = i_stat_itvl;
      ng = ng + 1; stat_itvl_anova_G{ng}(stat_itvl_anova_idat) = patient_absperttrial.subj(itrial); % don't use this in the anova
    end
    for itrial = 1:control_absperttrial.ntrials
      stat_itvl_anova_idat = stat_itvl_anova_idat + 1;
      stat_itvl_anova_devY(stat_itvl_anova_idat) = mean(control_absperttrial.segdevmean(itrial,iseg_lo:iseg_hi),2);
      ng = 0;
      ng = ng + 1; stat_itvl_anova_G{ng}(stat_itvl_anova_idat) = 'c';
      ng = ng + 1; stat_itvl_anova_G{ng}(stat_itvl_anova_idat) = i_stat_itvl;
      ng = ng + 1; stat_itvl_anova_G{ng}(stat_itvl_anova_idat) = control_absperttrial.subj(itrial); % don't use this in the anova
    end
  end
  ng2use = 2; % don't use subj ID in the anova
  stat_itvl_varnames = {'group' 'itvl'};

  figure
  
  [dev_stat_itvl.anova.p,dev_stat_itvl.anova.tab,dev_stat_itvl.anova.stats,dev_stat_itvl.anova.terms] = ...
      anovan(stat_itvl_anova_devY,stat_itvl_anova_G(1:ng2use),'varnames',stat_itvl_varnames,'model',2,'display','on'); % 'random',2);

  [dev_stat_itvl.multcomp.c,dev_stat_itvl.multcomp.m,dev_stat_itvl.multcomp.h,dev_stat_itvl.multcomp.combin_names] = ...
      multcompare(dev_stat_itvl.anova.stats,'Dimension',1:ng2use,'CType','bonferroni');
end

% individual subject means from these intervals:

group = {'patients','controls'};
ngroups = length(group);
group_id = {'p','c'};
ngood_subjs_in_group = [ngood_patients ngood_controls];
good_subj_idxes_in_group = {good_patient_idxes good_control_idxes};

for igroup = 1:ngroups
  group_idxes = find(stat_itvl_anova_G{1} == group_id{igroup});
  for isubj = 1:ngood_subjs_in_group(igroup)
    the_subj = good_subj_idxes_in_group{igroup}(isubj);
    subj_idxes = find(stat_itvl_anova_G{3} == the_subj);
    group_subj_idxes = intersect(group_idxes,subj_idxes);
    for i_stat_itvl = 1:n_stat_itvls
      itvl_idxes = find(stat_itvl_anova_G{2} == i_stat_itvl);
      group_subj_itvl_idxes = intersect(group_subj_idxes,itvl_idxes);
      seldat = stat_itvl_anova_devY(group_subj_itvl_idxes);
      n = length(seldat);
      m(isubj,i_stat_itvl) = mean(seldat);
      se = std(seldat)/sqrt(n);
      dev_stat_itvl.subjmeans.(group{igroup}).subj(isubj).n(i_stat_itvl) = n;
      dev_stat_itvl.subjmeans.(group{igroup}).subj(isubj).mean(i_stat_itvl) = m(isubj,i_stat_itvl);
      dev_stat_itvl.subjmeans.(group{igroup}).subj(isubj).stde(i_stat_itvl) = se;
    end
  end
  dev_stat_itvl.subjmeans.(group{igroup}).mean = mean(m);
  dev_stat_itvl.subjmeans.(group{igroup}).stde = std(m)/sqrt(ngood_subjs_in_group(igroup));
end

group_symbol = {'bo','ro'};
group_ploffset = [-0.1 0.1];
figure
hax = subplot(1,2,1);
for igroup = 1:ngroups
  for i_stat_itvl = 1:n_stat_itvls
    for isubj = 1:ngood_subjs_in_group(igroup)
      plot(i_stat_itvl+group_ploffset(igroup),dev_stat_itvl.subjmeans.(group{igroup}).subj(isubj).mean(i_stat_itvl),group_symbol{igroup});
      hold on
    end
  end
end
a = axis; axis([0 3 a(3:4)])
hold off
hax = subplot(1,2,2);
for igroup = 1:ngroups
  for i_stat_itvl = 1:n_stat_itvls
    m4pl = dev_stat_itvl.subjmeans.(group{igroup}).mean(i_stat_itvl);
    se4pl= dev_stat_itvl.subjmeans.(group{igroup}).stde(i_stat_itvl);
    plot(i_stat_itvl+group_ploffset(igroup),m4pl,group_symbol{igroup});
    hold on
    plot([i_stat_itvl i_stat_itvl]+group_ploffset(igroup),m4pl+[-se4pl se4pl],group_symbol{igroup}(1));
  end
end
a = axis; axis([0 3 a(3:4)])
hold off

anova_idat = 0;
for igroup = 1:ngroups
  for i_stat_itvl = 1:n_stat_itvls
    for isubj = 1:ngood_subjs_in_group(igroup)
      anova_idat = anova_idat + 1;
      stat_itvl_subjmeans_anova_devY(anova_idat) = dev_stat_itvl.subjmeans.(group{igroup}).subj(isubj).mean(i_stat_itvl);
      ng = 0;
      ng = ng + 1; stat_itvl_subjmeans_anova_G{ng}(anova_idat) = group_id{igroup};
      ng = ng + 1; stat_itvl_subjmeans_anova_G{ng}(anova_idat) = i_stat_itvl;
    end
  end
end
[dev_stat_itvl.subjmeans.anova.p, ...
 dev_stat_itvl.subjmeans.anova.tab, ...
 dev_stat_itvl.subjmeans.anova.stats, ...
 dev_stat_itvl.subjmeans.anova.terms] = ...
    anovan(stat_itvl_subjmeans_anova_devY,stat_itvl_subjmeans_anova_G, ...
           'varnames',stat_itvl_varnames,'model',2,'display','on'); % 'random',2);

fprintf('saving(dev_stat_itvl)...'); pause(0.2);
save('dev_stat_itvl','dev_stat_itvl');
fprintf('done\n');

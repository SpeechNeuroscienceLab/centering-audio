if ~exist('full_pert_resps','var')
  load('full_pert_resps');
end

yes_save_results = 1; % hardly ever want this zero (almost always want to save results)

centsdev_dat.tlims4centsbase = [-0.05 0.05];
% yes_check_good_bad = 1; % almost always want to check good/bad subject
yes_check_good_bad = 0; % only use this for finding good plots for paper

yes_plot_calc_centsdev = 1; % hardly ever want to plot this

centsdev_dat.frame_taxis = full_pert_resps.patients.subj(1).pert_resp.frame_taxis;
centsdev_dat.nframeswin  = full_pert_resps.patients.subj(1).pert_resp.nframeswin;

centsdev_dat.tlims4peak = [0.1 centsdev_dat.frame_taxis(end)];
centsdev_dat.framelims4peak = dsearchn(centsdev_dat.frame_taxis',centsdev_dat.tlims4peak')';
centsdev_dat.frameidxes4peak = centsdev_dat.framelims4peak(1):centsdev_dat.framelims4peak(2);

if yes_check_good_bad
  hf = figure % just for testing
end
group = {'patients' 'controls'};
ngroups = length(group);
for igroup = 1:ngroups
  fprintf('%s:\n',group{igroup});
  centsdev_dat.(group{igroup}).nsubj = full_pert_resps.(group{igroup}).nsubj;
  for isubj = 1:centsdev_dat.(group{igroup}).nsubj
    subj_title = sprintf('%s(%d)',group{igroup}(1:(end-1)),isubj);
    centsdev_dat.(group{igroup}).subj(isubj).path = full_pert_resps.(group{igroup}).subj(isubj).path;
    centsdev_dat.(group{igroup}).subj(isubj).date = full_pert_resps.(group{igroup}).subj(isubj).date;
%    centsdev_dat.(group{igroup}).subj(isubj).pidn = full_pert_resps.(group{igroup}).subj(isubj).pidn;
    fprintf('%s: %s\n',subj_title,centsdev_dat.(group{igroup}).subj(isubj).path);
    subj_n_good_trials = full_pert_resps.(group{igroup}).subj(isubj).pert_resp.n_good_trials;
    centsdev_dat.(group{igroup}).subj(isubj).n_good_trials = subj_n_good_trials;
    [the_dat the_mean4centsdev] = calc_centsdev_from_hzdat(full_pert_resps.(group{igroup}).subj(isubj).pert_resp, ...
                                                      centsdev_dat.tlims4centsbase,yes_plot_calc_centsdev,subj_title);
    centsdev_dat.(group{igroup}).subj(isubj).dat = the_dat;
    centsdev_dat.(group{igroup}).subj(isubj).mean4centsdev = the_mean4centsdev;
    centsdev_dat.(group{igroup}).subj(isubj).absdat = {-the_dat{1} the_dat{2} [-the_dat{1}; the_dat{2}]};
    for ipert = 1:3
      centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.dat = mean(centsdev_dat.(group{igroup}).subj(isubj).absdat{ipert},1);
      meandat4peak = centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.dat(centsdev_dat.frameidxes4peak);
      [peak,ipeak] = max(meandat4peak);
      centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.peak = peak;
      centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.iframe_peak = ipeak + centsdev_dat.framelims4peak(1) - 1;
    end
    centsdev_dat.(group{igroup}).is_follower(isubj) = 0;

    centsdev_dat.(group{igroup}).is_good(isubj) = 1;
    if yes_check_good_bad
      clf
      pl_color = {'r','g','b'};
      for ipert = 1:3
        dat2plot = centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.dat;
        hpl = plot(centsdev_dat.frame_taxis,dat2plot,pl_color{ipert});
        if ipert == 1
          title(sprintf('%s: ngood trials(%d,%d)',subj_title,subj_n_good_trials(1),subj_n_good_trials(2)));
          hold on; 
        end
      end
      hl = vline(centsdev_dat.tlims4peak(1),'k');
      still_checking = 1;
      while still_checking
        % note ipert should always = 3 at this point
        hl = vline(centsdev_dat.frame_taxis(centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.iframe_peak),'r');
        reply = input('good data? [y]/n/m: ','s');
        if isempty(reply), reply = 'y'; end
        switch(reply)
          case 'y'
            centsdev_dat.(group{igroup}).is_good(isubj) = 1;
            still_checking = 0;
          case 'n'
            centsdev_dat.(group{igroup}).is_good(isubj) = 0;
            still_checking = 0;
          case 'm'
            for ipert = 1:3
              meandat4peak = centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.dat(centsdev_dat.frameidxes4peak);
              [peak,ipeak] = min(meandat4peak);
              centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.peak = peak;
              centsdev_dat.(group{igroup}).subj(isubj).absmean{ipert}.iframe_peak = ipeak + centsdev_dat.framelims4peak(1) - 1;
            end
            centsdev_dat.(group{igroup}).is_follower(isubj) = 1;
            delete(hl)
        end
      end
    end
  end
end

if yes_save_results
  fprintf('saving(centsdev_dat)...'); pause(0.2);
  save('centsdev_dat','centsdev_dat');
  fprintf('done\n');
end


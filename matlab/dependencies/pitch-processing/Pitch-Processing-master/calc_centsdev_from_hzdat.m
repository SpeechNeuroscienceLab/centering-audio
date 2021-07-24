function [centsdev_dat mean4centsdev_out] = calc_centsdev_from_hzdat(pert_resp,tlims4centsbase,yes_plot,plot_title)
  
  if nargin < 2 || isempty(tlims4centsbase), tlims4centsbase = [-0.05 0.05]; end
  if nargin < 3 || isempty(yes_plot), yes_plot = 0; end
  if nargin < 4 || isempty(plot_title), plot_title = 'subject data'; end
  
  frame_taxis = pert_resp.frame_taxis;
  iframelims4centsbase = dsearchn(frame_taxis',tlims4centsbase')';
  idx4centsbase = iframelims4centsbase(1):iframelims4centsbase(2);

  % calc cents for subject's produced pitch
  for ipert = 1:3
    hzdat = pert_resp.pitch_in.dat{ipert};
    centsdat{ipert} = zeros(size(hzdat));
    n = pert_resp.n_good_trials(ipert);
    for itrial = 1:n
      hzdat4trial = hzdat(itrial,:);
      centsbase4trial = mean(hzdat(itrial,idx4centsbase),2);
      centsdat4trial = 1200*log2(hzdat4trial / centsbase4trial);
      centsdat{ipert}(itrial,:) = centsdat4trial;
    end
  end
    
  ipert4mean = 3;
  mean4centsdev = mean(centsdat{ipert4mean},1);
  
  for ipert = 1:3
    centsdev_dat{ipert} = zeros(size(centsdat{ipert}));
    n = pert_resp.n_good_trials(ipert);
    for itrial = 1:n
      centsdat4trial = centsdat{ipert}(itrial,:);
      centsdev_dat4trial = centsdat4trial - mean4centsdev;
      centsdev_dat{ipert}(itrial,:) = centsdev_dat4trial;
    end
  end

  if yes_plot

    t = frame_taxis;

    % calc cents for subject's feedback 
    for ipert = 1:2
      hzdat = pert_resp.pitch_in.dat{ipert};
      hzdat_out = pert_resp.pitch_out.dat{ipert};
      centsdat_out{ipert} = zeros(size(hzdat_out));
      n = pert_resp.n_good_trials(ipert);
      for itrial = 1:n
        hzdat_out4trial = hzdat_out(itrial,:);
        centsbase4trial = mean(hzdat(itrial,idx4centsbase),2); % calc'd from pitch_in, not pitch_out
        centsdat_out4trial = 1200*log2(hzdat_out4trial / centsbase4trial);
        centsdat_out{ipert}(itrial,:) = centsdat_out4trial;
      end
    end

    % calc mean and standard error (se) of everything to plot
    for ipert = 1:2
      n = pert_resp.n_good_trials(ipert);
      m_cents_in{ipert}  = mean(centsdat{ipert},1);
      se_cents_in{ipert} = std(centsdat{ipert},0,1)/sqrt(n);
      m_cents_out{ipert}  = mean(centsdat_out{ipert},1);
      se_cents_out{ipert} = std(centsdat_out{ipert},0,1)/sqrt(n);
      m_centsdev{ipert}  = mean(centsdev_dat{ipert},1);
      se_centsdev{ipert} = std(centsdev_dat{ipert},0,1)/sqrt(n);
    end
    n1 = pert_resp.n_good_trials(1);
    n2 = pert_resp.n_good_trials(2);
    abs_centsdev_dat = [-centsdev_dat{1};centsdev_dat{2}];
    m_abs_centsdev = mean(abs_centsdev_dat,1);
    se_abs_centsdev = std(abs_centsdev_dat,0,1)/sqrt(n1+n2); % n is correct here
    
    % plot feedback perturbations, and subject's responses, in cents
    hf = figure;
    set(hf,'Position',[155    51   560   819]);
    ihax = 0; nhax = 5;
    for ipert = 1:2
      ihax = ihax + 1; hax(ihax) = my_subplot(nhax,1,ihax);
      plot(t,m_cents_out{ipert},'b');
      hold on
      plot(t,m_cents_out{ipert}+se_cents_out{ipert},'b');
      plot(t,m_cents_out{ipert}-se_cents_out{ipert},'b');
      plot(t,m_cents_in{ipert},'r');
      hold on
      plot(t,m_cents_in{ipert}+se_cents_in{ipert},'r');
      plot(t,m_cents_in{ipert}-se_cents_in{ipert},'r');
      hl = hline(0,'k');
      switch ipert
        case 1
          ylabel('up pert');
          axis([t(1) t(end) -70 120]);
        case 2
          ylabel('down pert');
          axis([t(1) t(end) -120 70]);
      end
    end

    % overlay responses to each pert on the base (mean4centsdev) used to calc centsdev    
    ihax = ihax + 1; hax(ihax) = my_subplot(nhax,1,ihax);
    plot(t,mean4centsdev,'b');
    hold on
    for ipert = 1:2
      plot(t,m_cents_in{ipert},'r');
      hold on
      plot(t,m_cents_in{ipert}+se_cents_in{ipert},'r');
      plot(t,m_cents_in{ipert}-se_cents_in{ipert},'r');
    end
    ylabel('all trial mean');
    % a = axis; axis([t(1) t(end) a(3:4)]);
    axis([t(1) t(end) -60 60]);

    % plot response deviations (centsdev)
    ihax = ihax + 1; hax(ihax) = my_subplot(nhax,1,ihax);
    for ipert = 1:2
      plot(t,m_centsdev{ipert},'r');
      hold on
      plot(t,m_centsdev{ipert}+se_centsdev{ipert},'r');
      plot(t,m_centsdev{ipert}-se_centsdev{ipert},'r');
    end
    hl = hline(0,'k');
    ylabel('resp devs');
    % a = axis; axis([t(1) t(end) a(3:4)]);
    axis([t(1) t(end) -60 60]);
    
    ihax = ihax + 1; hax(ihax) = my_subplot(nhax,1,ihax);
    plot(t,m_abs_centsdev,'r');
    hold on
    plot(t,m_abs_centsdev+se_abs_centsdev,'r');
    plot(t,m_abs_centsdev-se_abs_centsdev,'r');
    hl = hline(0,'k');
    ylabel('comb resp');
    % a = axis; axis([t(1) t(end) a(3:4)]);
    axis([t(1) t(end) -10 60]);

    for ihax = 1:(nhax-1)
      axes(hax(ihax));
      set(hax(ihax),'XTick',[]);
      if ihax == 1, title(plot_title); end
    end
    axes(hax(nhax));
    xlabel('time since pert (sec)');
    
    

    
    % plot centsdev of responses to each pert, and plot base used to calc centsdev
    hf = figure;
    % set(hf,'Position',[1478         509         522         931]);
    ihax = 0;
    ipert = ipert4mean;        
    ihax = ihax + 1; hax(ihax) = subplot(3,1,ihax);
    plot(t,mean4centsdev,'b');
    hold on
    for ipert = 1:2
      ihax = ihax + 1; hax(ihax) = subplot(3,1,ihax);
      plot(t,m_centsdev{ipert},'b');
      hold on
      plot(t,m_centsdev{ipert}+se_centsdev{ipert},'b');
      plot(t,m_centsdev{ipert}-se_centsdev{ipert},'b');
      if ipert == 1
        ulim = max(m_centsdev{ipert}+se_centsdev{ipert});
        llim = min(m_centsdev{ipert}-se_centsdev{ipert});
      else
        if max(m_centsdev{ipert}+se_centsdev{ipert}) > ulim, ulim = max(m_centsdev{ipert}+se_centsdev{ipert}); end
        if min(m_centsdev{ipert}-se_centsdev{ipert}) < llim, llim = min(m_centsdev{ipert}-se_centsdev{ipert}); end
      end
    end
    ihax = 0;
    ihax = ihax + 1; axes(hax(ihax));
    title(plot_title);
    a = axis; axis([t(1) t(end) a(3:4)]);
    ylabel('centsbase');
    for ipert = 1:2
      ihax = ihax + 1; axes(hax(ihax));
      axis([t(1) t(end) llim ulim]);
      hl = hline(0,'k');
      ylabel('centsdev');
    end
    xlabel('time');
    
    pause;
  end

  if nargout >= 2, mean4centsdev_out = mean4centsdev; end
  
end % end of function

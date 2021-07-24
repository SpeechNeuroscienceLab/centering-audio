%{
    Newest make_baseline4comp.m 
        -Allows catch trials to be used as a baseline
    Edited by Yash Shroff 8/1/2019
%}    
function make_baseline4comp(pert_resp,colors,fig_pos)
npert_types = pert_resp.npert_types;
parsed_frame_taxis = pert_resp.frame_taxis;
 
reply = input('baseline type? [(l)inear]/(m)ean/ or (c)atch trial? response : ','s');
if isempty(reply) || strcmp(reply(1),'l')
  baseline_type = 'l';
elseif strcmp(reply(1), 'c')
  baseline_type = 'c';
else
  baseline_type = 'm';
end
 
mean_parsed_pitch_in =  pert_resp.cents.pitch_in.mean((npert_types+1),:);
stde_parsed_pitch_in =  pert_resp.cents.pitch_in.stde((npert_types+1),:);
mean_parsed_pitch_out = pert_resp.cents.pitch_out.mean((npert_types+1),:);
stde_parsed_pitch_out = pert_resp.cents.pitch_out.stde((npert_types+1),:);
 
meanresp_baseline = mean_parsed_pitch_in;
 
%For saving incase user doesn't choose linear
tlims4baseline = [];
linear_baseline = [];
catch_baseline = [];
 
if strcmp(baseline_type,'m')
  
  baseline = meanresp_baseline;
 save('baseline4comp','baseline','tlims4baseline','linear_baseline','meanresp_baseline','baseline_type'); 
elseif strcmp(baseline_type, 'l')
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
elseif strcmp(baseline_type, 'c') 
    
    load('parsed_pitch_trials_test.mat');
     
   
   
      
        load('pitch_trials.mat');
     
           
    good_catch_trials = zeros(length(good_trials),1);
    catch_baseline = zeros(length(good_trials),413);
      for itr = 1:length(good_trials)
        if (trial_pert_types(itr) == 0) 
          color_idx = rem(itr,ncolors); if ~color_idx, color_idx = ncolors; end
       hf = figure;
      axis([201 613 0 300]);
          hpl = plot(201:613,pitch_in(itr,201:613),colors{color_idx}); set(hpl,'Tag','pitch_in'); 
          reply = input('good trial? [y]/n: ','s'); %change default to yes
          if isempty (reply)
              good_catch_trials(itr) =1;
          
          elseif strcmp(reply,'y') 
              good_catch_trials(itr) =1;
          elseif strcmp(reply,'n') 
              good_catch_trials(itr) = 0;
          end
          
          
        
        end
        end
      
    
    
     

    n_good_catch_trials = nnz(good_catch_trials);
    for i = 1:1:length(good_catch_trials)
         
        if (good_catch_trials(i) == 1)
            catch_baseline(i,:) = pitch_in(i,201:613);
        end
         
    end
    catch_baseline(all(~catch_baseline,2),:) = [];
 
    baseline_hz = nanmean(catch_baseline);
    
    centsreference = nanmean(baseline_hz(:,1:34));
for i = 1:1:length(baseline_hz)
baseline_cents(:,i) = 1200*log2(baseline_hz(i)./centsreference);
end
baseline = baseline_cents;
save('baseline4comp','baseline','tlims4baseline','linear_baseline','meanresp_baseline','baseline_type', 'catch_baseline', 'n_good_catch_trials','good_catch_trials');
end


%{
function make_baseline4comp(pert_resp,colors,fig_pos)
 
npert_types = pert_resp.npert_types;
parsed_frame_taxis = pert_resp.frame_taxis;
 
reply = input('baseline type? [(l)inear]/(m)ean/ or (c)atch trial? response : ','s');
if isempty(reply) || strcmp(reply(1),'l')
  baseline_type = 'l';
elseif strcmp(reply(1), 'c')
  baseline_type = 'c';
else
  baseline_type = 'm';
end
 
mean_parsed_pitch_in =  pert_resp.cents.pitch_in.mean((npert_types+1),:);
stde_parsed_pitch_in =  pert_resp.cents.pitch_in.stde((npert_types+1),:);
mean_parsed_pitch_out = pert_resp.cents.pitch_out.mean((npert_types+1),:);
stde_parsed_pitch_out = pert_resp.cents.pitch_out.stde((npert_types+1),:);
 
meanresp_baseline = mean_parsed_pitch_in;
 
%For saving incase user doesn't choose linear
tlims4baseline = [];
linear_baseline = [];
 
if strcmp(baseline_type,'m')
  
  baseline = meanresp_baseline;
  
elseif strcmp(baseline_type, 'l')
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
elseif strcmp(baseline_type, 'c')   
    catch_pert_type = []
    for ipert_type = 1:length(pert_resp.pert_types)
        if pert_resp.pert_types(ipert_type) == 0
            catch_pert_type = ipert_type;
            break;
        end
    end
    %if user accidentaly selects catch but no catch trials were conducted the next line will result in baseline type m
    if isempty(catch_pert_type) catch_pert_type = npert_types + 1; disp('no catch found, using mean'); end
    catch_baseline =  pert_resp.cents.pitch_in.mean((catch_pert_type),:);
    baseline = catch_baseline;
end
save('baseline4comp','baseline','tlims4baseline','linear_baseline','meanresp_baseline','baseline_type'); 
%}	
%{
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
%}

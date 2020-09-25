function   make_parsed_pitch_trials(perttrial_pre,perttrial_dur,fs,ystep,ntrials,pitch_pert,pitch_out,pitch_in,ampl_out,ampl_in,wave_out,wave_in, ...
                                    pitchlimits,reject_pitchlimits,parse_fig_pos)

nframeswin  = round(perttrial_dur*fs/ystep);
nframes_pre = round(perttrial_pre*fs/ystep);
% iframe4pert = round(nframes/2);
iframe4pert = nframes_pre + 1;
iframe_low = iframe4pert - nframes_pre;
iframe_hi  = iframe_low + nframeswin - 1;
parsed_frame_taxis = (0:(nframeswin-1))*ystep/fs - perttrial_pre;
colors = {'m','c','r','g','b','k'};
ncolors = length(colors);
i_onsets = cell(ntrials,1);
i_offsets = cell(ntrials,1);
trial_pert_types = zeros(ntrials,1);
good_trials = ones(ntrials,1);
parsed_pitch_out = zeros(ntrials,nframeswin);
parsed_pitch_in  = zeros(ntrials,nframeswin);
parsed_ampl_out = zeros(ntrials,nframeswin);
parsed_ampl_in  = zeros(ntrials,nframeswin);


for itr = 1:ntrials
  i_onoffsets = find(diff(pitch_pert(itr,:)))+1;
  n_onoffsets = length(i_onoffsets);
  if ~n_onoffsets
      good_trials(itr) = 0;
      i_onsets{itr}  = -1; % i.e., not a valid array index
      i_offsets{itr} = -1; % i.e., not a valid array index
  else
      if rem(n_onoffsets,2)
          good_trials(itr) = 0;
          %error('n_onoffsets(%d) not even',n_onoffsets); Edited by Gill
          %01/16/2013
      end
      if n_onoffsets > 2, error('not currently setup to handle multiple perts per trial'); end
      n_onsets = n_onoffsets/2;
      i_onsets{itr}  = i_onoffsets(1:2:(n_onoffsets-1));
      i_offsets{itr} = i_onoffsets(2:2:(n_onoffsets));
  end
end
for itr = 1:ntrials
  if good_trials(itr)
    trial_pert_types(itr) = pitch_pert(itr,round((i_onsets{itr} + i_offsets{itr})/2));
  end
end
for itr = 1:ntrials
  if good_trials(itr)
    adj_pitch_temp = adj2onset(pitch_out(itr,:), iframe4pert,i_onsets{itr});
    parsed_pitch_out(itr,:) = adj_pitch_temp(iframe_low:iframe_hi);
    adj_pitch_temp = adj2onset(pitch_in(itr,:), iframe4pert,i_onsets{itr});
    parsed_pitch_in(itr,:)  = adj_pitch_temp(iframe_low:iframe_hi);
    adj_ampl_temp = adj2onset(ampl_out(itr,:), iframe4pert,i_onsets{itr});
    parsed_ampl_out(itr,:) = adj_ampl_temp(iframe_low:iframe_hi);
    adj_ampl_temp = adj2onset(ampl_in(itr,:), iframe4pert,i_onsets{itr});
    parsed_ampl_in(itr,:)  = adj_ampl_temp(iframe_low:iframe_hi);
  end
end
ibaselims = dsearchn(parsed_frame_taxis',[-0.5, -0.25]');
inoiselims = dsearchn(parsed_frame_taxis',[0.15, 1.18]');

hf = figure;
set(hf,'Position',parse_fig_pos);
while 1
  clf
  hax1 = subplot(211)
  hax2 = subplot(212)
  x = 0;
  for itr = 1:ntrials
    if good_trials(itr)
      color_idx = rem(itr,ncolors); if ~color_idx, color_idx = ncolors; end
      axes(hax1);
      hpl = plot(parsed_frame_taxis,parsed_pitch_out(itr,:),colors{color_idx}); set(hpl,'Tag','pitch_out'); set(hpl,'Userdata',itr);
      hold on
      axis([parsed_frame_taxis(1) parsed_frame_taxis(end) pitchlimits]);
      axes(hax2);
      hpl = plot(parsed_frame_taxis,parsed_pitch_in(itr,:),colors{color_idx}); set(hpl,'Tag','pitch_in'); set(hpl,'Userdata',itr);
      hold on
      axis([parsed_frame_taxis(1) parsed_frame_taxis(end) pitchlimits]);
      x = x+1;
      disp(x)
    end
  end
  %
  reply = input('delete a range of trials? [y]/n: ','s'); %change default to yes
  if strcmp(reply,'n'), break; end
  fprintf('select the y-coordinate from which you would like to delete\n');
  
  [xx,yy] = ginput(1);
  cur_obj = get(hf,'CurrentObject');
  itrial2del = get(cur_obj,'Userdata')
  while 1
    reply2 = input('delete trials above or below, a/b: ', 's');
    switch(reply2) 
        case 'a' 
            reply = input('delete all trials at or above this? [y]/n/p: ','s');
            if isempty(reply), reply = 'y'; end
            switch reply
              case 'y'
                for trial = 1:size(parsed_pitch_out, 1)
                    curr = parsed_pitch_out(trial,:);
                    m = max(curr);
                    if(m >= yy), good_trials(trial) = 0;end
                end
                break;
              case 'n', good_trials(itrial2del) = 1; delete(hpl); break;
              case 'p', soundsc(wave_out(itrial2del,:),fs); soundsc(wave_in(itrial2del,:),fs);
            end
        case 'b' 
            reply = input('delete all trials below this? [y]/n/p: ','s')   
            if isempty(reply), reply = 'y'; end
            switch reply
              case 'y'
                for trial = 1:size(parsed_pitch_out, 1)
                    curr = parsed_pitch_out(trial,:);
                    m = min(curr);
                    if(m <= yy), good_trials(trial) = 0;end
                end
                break;
              case 'n', good_trials(itrial2del) = 1; delete(hpl); break;
              case 'p', soundsc(wave_out(itrial2del,:),fs); soundsc(wave_in(itrial2del,:),fs);
            end
    end
  end
end
while 1
    clf
  hax1 = subplot(211);
  hax2 = subplot(212);
  for itr = 1:ntrials
    if good_trials(itr)
      color_idx = rem(itr,ncolors); if ~color_idx, color_idx = ncolors; end
      axes(hax1);
      hpl = plot(parsed_frame_taxis,parsed_pitch_out(itr,:),colors{color_idx}); set(hpl,'Tag','pitch_out'); set(hpl,'Userdata',itr);
      hold on
      axis([parsed_frame_taxis(1) parsed_frame_taxis(end) pitchlimits]);
      axes(hax2);
      hpl = plot(parsed_frame_taxis,parsed_pitch_in(itr,:),colors{color_idx}); set(hpl,'Tag','pitch_in'); set(hpl,'Userdata',itr);
      hold on
      axis([parsed_frame_taxis(1) parsed_frame_taxis(end) pitchlimits]);
    end
  end
  reply = input('delete any trials? [y]/n: ','s'); %change default to yes
  if strcmp(reply,'n'), break; end
  fprintf('pick trial to delete\n');
  
  [xx,yy] = ginput(1);
  cur_obj = get(hf,'CurrentObject');
  itrial2del = get(cur_obj,'Userdata')
  switch(get(cur_obj,'Tag'))
    case 'pitch_out'
      axes(hax1);
      hpl = plot(parsed_frame_taxis,parsed_pitch_out(itrial2del,:),'r'); set(hpl,'LineWidth',3);
    case 'pitch_in'
      axes(hax2);
      hpl = plot(parsed_frame_taxis,parsed_pitch_in(itrial2del,:),'r'); set(hpl,'LineWidth',3);
  end
  while 1
    reply = input('delete this trial? [y]/n/p: ','s');
    if isempty(reply), reply = 'y'; end
    switch reply
      case 'y', good_trials(itrial2del) = 0; break;
      case 'n', good_trials(itrial2del) = 1; delete(hpl); break;
      case 'p', soundsc(wave_out(itrial2del,:),fs); soundsc(wave_in(itrial2del,:),fs);
    end
  end
end  
close(hf);

save('parsed_pitch_trials_test','nframeswin','nframes_pre','iframe4pert','iframe_low','iframe_hi', ...
     'parsed_frame_taxis','colors','ncolors','i_onsets','i_offsets','trial_pert_types','good_trials','parsed_pitch_out','parsed_pitch_in');

function [vods,pitchlimits, perttrial_pre, perttrial_dur, pertclass_pitch_types, yes_overwrite] = get_init_info(subj_info)

init_info_file = './pitch_anal_init_info.mat';

curdir = cd;
%cd(rootdir);
%cd(subj_info(1).subjdir);
%cd(subj_info(1).speak_consolidate_audiodir);

if ~exist(init_info_file,'file')
%   pitchlimits = [50 300];
%   perttrial_pre = 0.45; perttrial_dur = 1.6;
%   pertclass_pitch_types = [100 -100];
  yes_overwrite = 0; % set to 6 for complete reprocessing, or 1 for final step (xcspec) reprocessing, or 0 to just reprint response plots
else
  load(init_info_file);
end
vods=input(sprintf('Was pitch perturbation triggered relative to voice onset? [Y]/N : '),'s');
if isempty(vods) 
    vods = 'Y';
end

prompt={'Pitch lower limit: ','Pitch upper limit: ','Pre-perturbation time: ', 'Duration of trial: ','Pitch pert values: '};
title='Pitch Perturbation Parameters ';
num_lines=1;
if ~exist(init_info_file,'file')
def={'50','300','0.2','1.2',['100' ' -100']};
else
def ={num2str(pitchlimits(1)),num2str(pitchlimits(2)),num2str(perttrial_pre),num2str(perttrial_dur),num2str(pertclass_pitch_types)};
end
reply=inputdlg(prompt,title,num_lines,def,'on');
pitchlimits = [str2num(reply{1}),str2num(reply{2})];
  perttrial_pre = str2num(reply{3}); 
  perttrial_dur = str2num(reply{4});
  pertclass_pitch_types = str2num(reply{5});
fprintf('set overwrite level to 6 for complete reprocessing, or 1 for final step (xcspec) reprocessing, or 0 to just reprint response plots\n');
reply = input(sprintf('use overwrite level(%d)? [y]/n: ',yes_overwrite),'s');
if ~isempty(reply) && ~strcmp(reply,'y')
  yes_overwrite = input('overwrite level: ');
end
save(init_info_file,'pitchlimits','perttrial_pre','perttrial_dur','pertclass_pitch_types','yes_overwrite');

cd(curdir);


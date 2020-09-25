cd('speak_consolidate_audiodir/');
load('pert_resp.mat');
tpeak = (mean(pert_resp.tpeak{1,1})+mean(pert_resp.tpeak{1,2}))/2;
peakmag = (abs(pert_resp.xcspec_mean{1,1}.peak)+abs(pert_resp.xcspec_mean{1,2}.peak))/2;

%For uni-directional shift
% tpeak = mean(pert_resp.tpeak{1,1});
% peakmag = abs(pert_resp.xcspec_mean{1,1}.peak);

disp(tpeak);
disp(peakmag);
disp(pert_resp.n_good_trials);
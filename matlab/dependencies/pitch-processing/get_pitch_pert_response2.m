function pert_resp = get_pitch_pert_response2(trial_pert_types,parsed_pitch_in,parsed_pitch_out,good_trials,parsed_frame_taxis,cents_refpoint_t,pert_types)
%function pert_resp = get_pitch_pert_response2(trial_pert_types,parsed_pitch_in,parsed_pitch_out,good_trials,parsed_frame_taxis,pert_types)

if nargin < 6 || isempty(cents_refpoint_t), cents_refpoint_t = 0; end
if nargin < 7 pert_types = []; end

pert_type_list = unique(trial_pert_types(logical(good_trials)));
if isempty(pert_types)
  pert_resp.pert_types = pert_type_list';
else
  len_pert_type_list = length(pert_type_list);
  len_pert_types = length(pert_types);
  error_if_unequal(len_pert_types,len_pert_type_list,'%d');
  if any(abs(pert_type_list' - sort(pert_types)) ~= 0)
    error('pert_types mismatch');
  end
  pert_resp.pert_types = pert_types;
end
pert_resp.npert_types = length(pert_resp.pert_types);
for ipert_type = 1:(pert_resp.npert_types + 1) % last "pert_type" is resp to all perts, regardless of type
  if ipert_type > pert_resp.npert_types
    pert_resp.good_trials{ipert_type} = sort([pert_resp.good_trials{1:(ipert_type-1)}]);
  else
    pert_resp.good_trials{ipert_type} = find((trial_pert_types == pert_resp.pert_types(ipert_type)) & good_trials)';
  end
  pert_resp.n_good_trials(ipert_type) = length(pert_resp.good_trials{ipert_type});
  pert_resp.pitch_in.dat{ipert_type} = parsed_pitch_in(pert_resp.good_trials{ipert_type},:);
  pert_resp.pitch_in.mean(ipert_type,:) = mean(pert_resp.pitch_in.dat{ipert_type},1);
  pert_resp.pitch_in.stde(ipert_type,:) = std(pert_resp.pitch_in.dat{ipert_type},0,1)/sqrt(pert_resp.n_good_trials(ipert_type));
  pert_resp.pitch_out.dat{ipert_type} = parsed_pitch_out(pert_resp.good_trials{ipert_type},:);
  pert_resp.pitch_out.mean(ipert_type,:) = mean(pert_resp.pitch_out.dat{ipert_type},1);
  pert_resp.pitch_out.stde(ipert_type,:) = std(pert_resp.pitch_out.dat{ipert_type},0,1)/sqrt(pert_resp.n_good_trials(ipert_type));
end

pert_resp.frame_taxis = parsed_frame_taxis;
nframeswin = length(parsed_frame_taxis);
pert_resp.nframeswin = nframeswin;

pert_resp.cents.refpoint.t = cents_refpoint_t; % seconds
pert_resp.cents.refpoint.i = dsearchn(pert_resp.frame_taxis',pert_resp.cents.refpoint.t);
% pert_resp.cents.refpoint.n = dsearchn(pert_resp.frame_taxis',0);
% pert_resp.cents.refpoint.x = median(median(pert_resp.pitch_out.dat{1,3}(:,1:pert_resp.cents.refpoint.n)))
% pert_resp.cents.refpoint.t = dsearchn(pert_resp.pitch_out.mean(1,:)',pert_resp.cents.refpoint.x); % seconds
% pert_resp.cents.refpoint.i = dsearchn((pert_resp.frame_taxis(:,1:pert_resp.cents.refpoint.n))',pert_resp.cents.refpoint.t);
pert_resp.cents.centsfact = 1200/log(2);
npert_types = pert_resp.npert_types;
for ipert_type = 1:(npert_types+1)
  pert_resp.cents.pitch_in.refpoint.dat{ipert_type} = pert_resp.pitch_in.dat{ipert_type}(:,pert_resp.cents.refpoint.i);
  refpoint_repmat = repmat(pert_resp.cents.pitch_in.refpoint.dat{ipert_type},[1 nframeswin]);
  pert_resp.cents.pitch_in.dat{ipert_type} = pert_resp.cents.centsfact*log(pert_resp.pitch_in.dat{ipert_type} ./ refpoint_repmat);
  pert_resp.cents.pitch_in.mean(ipert_type,:) = mean(pert_resp.cents.pitch_in.dat{ipert_type},1);
  pert_resp.cents.pitch_in.stde(ipert_type,:) = std(pert_resp.cents.pitch_in.dat{ipert_type},0,1)/sqrt(pert_resp.n_good_trials(ipert_type));

  pert_resp.cents.pitch_out.refpoint.dat{ipert_type} = pert_resp.pitch_out.dat{ipert_type}(:,pert_resp.cents.refpoint.i);
  refpoint_repmat = repmat(pert_resp.cents.pitch_out.refpoint.dat{ipert_type},[1 nframeswin]);
  pert_resp.cents.pitch_out.dat{ipert_type} = pert_resp.cents.centsfact*log(pert_resp.pitch_out.dat{ipert_type} ./ refpoint_repmat);
  pert_resp.cents.pitch_out.mean(ipert_type,:) = mean(pert_resp.cents.pitch_out.dat{ipert_type},1);
  pert_resp.cents.pitch_out.stde(ipert_type,:) = std(pert_resp.cents.pitch_out.dat{ipert_type},0,1)/sqrt(pert_resp.n_good_trials(ipert_type));
end

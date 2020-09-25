

yes_reload = 0;
if yes_reload || ~exist('num','var')
  clear all
  subj_info_file = 'svPPA_info.xls';

  [num,txt,raw] = xlsread(subj_info_file);
  [nrows_raw,ncols_raw] = size(raw);
end

icol4date = strmatch('Date',txt(1,:));
icol4order = strmatch('expr_order',txt(1,:));
icol4type = strmatch('Type',txt(1,:));
icol4comp = strmatch('mean_comp',txt(1,:));
icol4stde = strmatch('stde_comp',txt(1,:));
%icol4pidn = strmatch('PIDN',txt(1,:));
icol4include = strmatch('include_in_study',txt(1,:));

if ~ispc
    datecolumn = raw(:,icol4date);
    numbervalues = cellfun(@(V) any(isnumeric(V(:)) && ~isnan(V(:))), datecolumn)';
    datedata = datecolumn(numbervalues,1);
    correcteddatedata = cellfun(@(V) V-1+693961, datedata);
    converteddatedata = datestr(correcteddatedata,'mm/dd/yyyy');
    cellsconverteddatedata = cellstr(converteddatedata);
    
    firstdatesindices = find(numbervalues == 1,1,'first');
    lastdatesindices = find(numbervalues == 1,1,'last');
    datecolumn(firstdatesindices:lastdatesindices,1) = cellsconverteddatedata;
    raw(:,icol4date) = datecolumn;
    [row_w col_l] = size(txt);
    txt(:,icol4date) = datecolumn(1:row_w,1);
end

for irow = 1:nrows_raw
  if isnumeric(raw{irow,icol4comp})
    compdat_from_xls(irow) = raw{irow,icol4comp};
    if ~isnumeric(raw{irow,icol4stde}), error('should have stde for comp'); end
    compstde_from_xls(irow) = raw{irow,icol4stde};
  else
    compdat_from_xls(irow) = NaN;
    compstde_from_xls(irow) = NaN;
  end
end  
    

irows4comp = find(~isnan(compdat_from_xls));
nrows4comp = length(irows4comp);

irows4include = strmatch('yes',txt(:,icol4include));

irows4all_patients = strmatch('patient',txt(:,icol4type));
irows4all_controls = strmatch('control',txt(:,icol4type));

irows4patients = intersect(irows4all_patients,intersect(irows4include,irows4comp));
nrows4patients = length(irows4patients);
irows4controls = intersect(irows4all_controls,intersect(irows4include,irows4comp));
nrows4controls = length(irows4controls);

patient_comp(:,1) = compdat_from_xls(irows4patients);
patient_comp(:,2) = compstde_from_xls(irows4patients);

control_comp(:,1) = compdat_from_xls(irows4controls);
control_comp(:,2) = compstde_from_xls(irows4controls);

curdir = cd;
fprintf('patients:\n');
cd('Patients');
patients_dir = cd;
for isubj = 1:nrows4patients
  [the_expr_dir,the_expr_date] = date_to_dir3(txt,raw,irows4patients,icol4date,icol4order,isubj);
  cd(the_expr_dir);
  the_pert_resp_file = get_pert_resp_file();
  clear subj;
  fprintf('load(%s)...',the_pert_resp_file); pause(0.2);
  subj = load(the_pert_resp_file);
  fprintf('done\n');
  subj.path = the_pert_resp_file;
  subj.date = the_expr_date;
 % subj.pidn = raw{irows4patients(isubj),icol4pidn};
  full_pert_resps.patients.subj(isubj) = subj;
  cd(patients_dir);
end
full_pert_resps.patients.nsubj = isubj;
cd(curdir)

cd('Controls');
controls_dir = cd;
for isubj = 1:nrows4controls
  [the_expr_dir,the_expr_date] = date_to_dir3(txt,raw,irows4controls,icol4date,icol4order,isubj);
  cd(the_expr_dir);
  the_pert_resp_file = get_pert_resp_file();
  clear subj;
  fprintf('load(%s)...',the_pert_resp_file); pause(0.2);
  subj = load(the_pert_resp_file);
  fprintf('done\n');
  subj.path = the_pert_resp_file;
  subj.date = the_expr_date;
  %subj.pidn = raw{irows4controls(isubj),icol4pidn};
  full_pert_resps.controls.subj(isubj) = subj;
  cd(controls_dir);
end
full_pert_resps.controls.nsubj = isubj;
cd(curdir)
fprintf('saving(full_pert_resps)...'); pause(0.2);
save('full_pert_resps','full_pert_resps');
fprintf('done\n');

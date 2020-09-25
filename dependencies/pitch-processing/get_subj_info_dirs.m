
function subj_info = get_subj_info_dirs(speak_consolidate_audiodir)

disp('Select folder with dir (i.e. 20140314) and file with data');
onlyfoldersfilter = {'*.unwantedfiletype', 'fake non-existent filetype'};
foldersselected = uipickfiles('Type',onlyfoldersfilter);

lengthoffoldernames= length(foldersselected);
if ispc 
    slash_char = '\'
else
    slash_char = '/'
end
if lengthoffoldernames ~=0 && iscell(foldersselected)
    for loop = 1:lengthoffoldernames
    if ispc
            indofslash = strfind(foldersselected{1,loop},'\');
        else
            indofslash = strfind(foldersselected{1,loop},'/');
        end
        lastslash = indofslash(end) + 1;
        subjdir = foldersselected{1,loop}((lastslash):end);
        rootdir = foldersselected{1,loop}(1:lastslash-1);
        cd(rootdir);
        cd(subjdir);
% GUI is used to select specific file with data
exprdirs = uipickfiles('Type', onlyfoldersfilter);
d = exprdirs;
nd = length(exprdirs);75
nexprdirs = nd;
if ~nexprdirs, error('no exprdirs found'); 
end
                                               
naudiodirs = 0;
curdir = cd;
for iexprdir = 1:nexprdirs
  exprdir = exprdirs{iexprdir};
  cd(exprdir);
  cd speak
  d = dir('block*');
  nblockdirs = sum([d.isdir]); % don't want to count block.done file
  for iblockdir = 1:nblockdirs
    naudiodirs = naudiodirs + 1;
    speak_audiodirs{naudiodirs,1} = [exprdir slash_char 'speak' slash_char sprintf('block%d',iblockdir-1)];
  end
  cd(curdir)
end

if isempty(dir(speak_consolidate_audiodir))
  mkdir(speak_consolidate_audiodir);
end

subj_info(1).subjdir = subjdir;
subj_info(1).speak_audiodirs = speak_audiodirs;
subj_info(1).speak_consolidate_audiodir = speak_consolidate_audiodir;
subj_info(1).exprdirs = exprdirs;
    end
end
%cd(orig_dir)
%}

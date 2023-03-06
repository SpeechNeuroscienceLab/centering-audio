function make_consolidated_pitch_trials(consolidate_audiodir,speak_audiodirs,pitchlimits)

curdir_subj = cd;
nblocks = length(speak_audiodirs);
% for iblock = 1:nblocks
%   expr_audiodir = speak_audiodirs{iblock};
%   cd(expr_audiodir); fprintf('cd expr_audiodir(%s)\n',expr_audiodir);
%   if ~exist('pitch_trials.mat','file'), make_pitch_trials2(pitchlimits,0); end
%   cd(curdir_subj);
% end
for iblock = 1:nblocks
  expr_audiodir = speak_audiodirs{iblock};
  cd(expr_audiodir); fprintf('cd expr_audiodir(%s)\n',expr_audiodir);
  if ~exist('pitch_trials.mat','file'), error('pitch_trials.mat not found'); end
  pitch_trials4block = load('pitch_trials');
  if iblock == 1
    consld_pitch_trials = pitch_trials4block;
    consld_pitch_trials.iblock = iblock*ones(pitch_trials4block.ntrials,1);
    consld_pitch_trials.itrial_in_block = (1:(pitch_trials4block.ntrials))';
    fldname = fieldnames(pitch_trials4block);
    nfldnames = length(fldname);
  else
    for ifldname = 1:nfldnames
      the_fldname = fldname{ifldname};
      switch the_fldname
        case 'ntrials'
          consld_pitch_trials.ntrials = consld_pitch_trials.ntrials + pitch_trials4block.ntrials;
          consld_pitch_trials.iblock = [consld_pitch_trials.iblock; iblock*ones(pitch_trials4block.ntrials,1)];
          consld_pitch_trials.itrial_in_block = [consld_pitch_trials.itrial_in_block; (1:(pitch_trials4block.ntrials))'];
        otherwise
          if size(pitch_trials4block.(the_fldname),1) > 1
            consld_pitch_trials.(the_fldname) = [consld_pitch_trials.(the_fldname); pitch_trials4block.(the_fldname)];
          else
            if any(consld_pitch_trials.(the_fldname) ~= pitch_trials4block.(the_fldname))
              error('mismatch in (%s) at block(%d)',the_fldname,iblock);
            end
          end
      end
    end
  end
  cd(curdir_subj);
end
cd(consolidate_audiodir)
save('pitch_trials','-struct','consld_pitch_trials');
cd(curdir_subj)

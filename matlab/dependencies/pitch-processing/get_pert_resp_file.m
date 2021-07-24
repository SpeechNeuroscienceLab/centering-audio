function the_pert_resp_file = get_pert_resp_file()
mainfolder = pwd;
% consolidate_block_found = 0;
% exprdirs = dir('expr*');
% for iexprdir = 1:length(exprdirs)
%   if exprdirs(iexprdir).isdir
%     exprsubdirs = dir([exprdirs(iexprdir).name '/speak']);
%     for iexprsubdir = 1:length(exprsubdirs)
%       if exprsubdirs(iexprsubdir).isdir && strcmp(exprsubdirs(iexprsubdir).name,'consolidate_block')
%         consolidate_block_found = 1;
%         break;
%       end
%     end
%   end
%   if consolidate_block_found
%     break;
%   end
%        
% end

%if ~consolidate_block_found
      the_pert_resp_file = sprintf('%s/speak_consolidate_audiodir/pert_resp.mat',mainfolder);
%end
  
% if consolidate_block_found
% fprintf('consolidate_block_found(%d)\n',consolidate_block_found);
% the_pert_resp_file = sprintf('%s/speak/consolidate_block/pert_resp.mat',exprdirs(iexprdir).name);
% end

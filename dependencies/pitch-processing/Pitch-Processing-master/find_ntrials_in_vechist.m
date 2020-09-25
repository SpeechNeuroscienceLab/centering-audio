function ntrials = find_ntrials_in_vechist(vechist)
% function ntrials = find_ntrials_in_vechist(vechist)
% finds ntrials by looking for discontinuities in iframe


yes_iframe_advance = diff(vechist.iframe(:,1)) > 0;
longest_iframe_advance = min(find(yes_iframe_advance ~= 1));
if isempty(longest_iframe_advance)
  ntrials = vechist.ntrials;
else
  ntrials = longest_iframe_advance;
end

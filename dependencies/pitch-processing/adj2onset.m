function adj_sig = adj2onset(sig,i4onset,i_onset)
% function adj_sig = adj2onset(sig,i4onset,i_onset)
% i4onset = desired position of onset in sig
% i_onset = actual  position of onset in sig
% adj_sig has onset at position i4onset, with appropriate zero padding

n = length(sig);
n2adj = i4onset - i_onset;
if n2adj >= 0
  adj_sig(1:n2adj) = 0;
  adj_sig((1 + n2adj):n) = sig(1:(n - n2adj));
else
  adj_sig(1:(n + n2adj)) = sig((1 - n2adj):n);
  adj_sig((n + n2adj + 1):n) = 0;
end

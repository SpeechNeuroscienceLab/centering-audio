function TS = threshspec(S,th)
% function TS = threshspec(S,th)

subS = S - th;
absubS = abs(subS);
TS = 0.5*(absubS + subS);

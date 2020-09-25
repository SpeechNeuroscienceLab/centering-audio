function x_mod = my_mod(x,m)
% function x_mod = my_mod(x,m)

x_mod = rem(x,m);
if ~x_mod && x
  x_mod = m;
end

    

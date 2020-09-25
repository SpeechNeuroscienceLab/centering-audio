function error_if_unequal(var1,var2,numfmt_str)

varname1 = inputname(1);
varname2 = inputname(2);

err_fmtstr = sprintf('%s(%s) ~= %s(%s)',varname1,numfmt_str,varname2,numfmt_str);

if var1 ~= var2
  error(sprintf(err_fmtstr,var1,var2));
end

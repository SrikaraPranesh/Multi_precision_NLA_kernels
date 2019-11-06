function [x] = ip_chop(a,b,format)
%IP_CHOP computes inner product using chop function
%   Note -- Requires chop.m to be in the MATLAB path
if nargin<3, format = 'h'; end
fp.format = format; chop([],fp);

n = length(a);
x = 0;
for i=1:n
    x = chop(x+chop(a(i,1)*b(i,1)));
end

end


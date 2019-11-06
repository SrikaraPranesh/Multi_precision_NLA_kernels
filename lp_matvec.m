function c = lp_matvec(A,b,format)
%LP_MATVEC low precision matrix-vector product
%   Performs matrix-vector multiplication in low precision format.
%   By default the low precision format is fp16
%   Note -- Requires 'chop' to be in the MATLAB path.

if nargin < 3
    fp.format = 'h'; chop([],fp);
else
    fp.format = format; chop([],fp);
end

% convert input data to low preision format
A = chop(A);
[m,~] = size(A);
b = chop(b);
c = zeros(m,1);

for i = 1:length(b)
    c = chop(c+chop(b(i,1)*A(:,i)));
end


end


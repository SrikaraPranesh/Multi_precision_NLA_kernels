function Ch = hgemm(A,B,format)
%hgemm Matrix-Matrix multiplication in half precision.
%   Note -- chop.m must be in the matlab path
if nargin < 3
    fp.format = 'h'; chop([],fp);
else
    fp.format = format; chop([],fp);
end

% double check if the input matrices are in the
%   required precision
A = chop(A); B = chop(B);

[m,k] = size(A);
[k,n] = size(B);
Ch = zeros(m,n);

for i = 1:k
   Ch = chop(Ch + (chop(chop(A(:,i))*chop(B(i,:))))); 
end

end


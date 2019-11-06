function x = trisol(T,b,format)
%TRISOL   Solve triangualr system in low precision arithmetic.
%   TRISOL(T, b) is the solution x to the triangular system
%   Tx = b computed in low precision arithmetic.
%   By default the low precision format is fp16
%   It requires the CHOP function to simulate lower precision arithmetic.

if nargin < 3
    fp.format = 'h'; chop([],fp);
else
    fp.format = format; chop([],fp);
end

% Convert matrix and vector to required precisions
T = chop(T); b = chop(b);

n = length(T);
x = zeros(n,1);

if ~norm(T-triu(T),1)      % Upper triangular

    x(n) = chop( b(n)/T(n,n));
    for i=n-1:-1:1
        temp = chop( x(i+1) .* T(1:i,i+1));
        b(1:i) = chop( b(1:i) - temp);
        x(i) = chop( b(i)/T(i,i));
    end

elseif ~norm(T-tril(T),1)   % Lower triangular

    x(1) = chop( b(1)/T(1,1));
    for i=2:n
        temp = chop( x(i-1) .* T(i:n,i-1));
        b(i:n) = chop( b(i:n) - temp);
        x(i) = chop(b(i)/T(i,i));
    end

else
   error('Matrix T must be triangular.')
end

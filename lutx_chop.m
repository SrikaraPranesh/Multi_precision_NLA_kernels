function [L,U,p] = lutx_chop(A,format)
%LUTX_CHOP Triangular factorization, textbook version [L,U,p] =
%   lutx_xhop(A) produces a unit lower triangular matrix L, an upper
%   triangular matrix U, and a permutation vector p, so that L*U = A(p,:)
%   It requires the CHOP function to simulate lower precision arithmetic.
%   By default half precision is used

if nargin<2, format = 'h'; end
fp.format = format; chop([],fp);

[m,n] = size(A);
p = (1:m)';

for k = 1:n-1

   % Find index of largest element below diagonal in k-th column
   [~,ind] = max(abs(A(k:n,k)));
   ind = ind+k-1;

   % Skip elimination if column is zero
   if (A(ind,k) ~= 0)
   
      % Swap pivot row
      if (ind ~= k)
         A([k ind],:) = A([ind k],:);
         p([k ind]) = p([ind k]);
      end

      % Compute multipliers
      i = k+1:m;
      A(i,k) = chop(A(i,k)/A(k,k));

      % Update the remainder of the matrix
      j = k+1:n;
      A(i,j) = chop(A(i,j) - chop(A(i,k)*A(k,j))); 
   end
end

% Separate result
L = tril(A,-1) + eye(m,n);
U = triu(A);U = U(1:n,1:end);
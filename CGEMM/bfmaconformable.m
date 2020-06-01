function [conformable, sizes] = bfmaconformable(A,B,C)
% BFMACONFORMABLE Check sizes of matrices for block fused multiply-add.

  [m,p] = size(A);
  [p1,n] = size(B);
  if nargin == 3
    [m1,n1] = size(C);
  end

  if p == p1 && (nargin == 2 || (m == m1 && n == n1))
    conformable = true;
    sizes = [m,p,n];
  else
    conformable = false;
    sizes = [];
  end

end
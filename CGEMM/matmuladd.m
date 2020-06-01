function C = matmuladd(A,B,C,bfma,use_chop)
%  Matrix vector product with simulated block fused multiply-add.
%   C = MATMULADD(A,BC,) computes the matrix A * B + C, where A, B, and C are
%   an M-by-P, P-by-N, and M-by-N matrix, respectively. The computation is
%   performed simulating the use of a block fused multiply-add kernel that
%   computes the product of two matrices of size BFMA(1)-by-BFMA(2) and
%   BFMA(2)-by-BFMA(3) and adds them to a matrix of size BFMA(1)-by-BFMA(3).
%
%   C = MATMULADD(A,B,BFMA,USE_CHOP) does not rely on CHOP if the Boolean
%   variable USE_CHOP is FALSE, and is equivalent to MATMULADD(A,B) otherwise.
%
%   For performance reasons, this code assumes that the matrices have sizes
%   multiple of the corresponding entry in BFMA. This can be achieved by
%   running [A,B,C] = BFMAPADDING(A,B,BFMA,C).

%   Note: We are assming that double precision does not incurr double rounding
%   in half precision.

  if nargin < 4
    bfma = [4,4,4];
  end

  if nargin < 5 || use_chop
    A = double(chop(A));
    B = double(chop(B));
    C = double(C);
  end

  [conformable, sizes] = bfmaconformable(A,B,C);
  assert(conformable)

  % Check that matrix dimensions are mutiple of four.
  assert(mod(sizes(1),bfma(1)) == 0)
  assert(mod(sizes(2),bfma(2)) == 0)
  assert(mod(sizes(3),bfma(3)) == 0)
  m4 = sizes(1) / bfma(1);
  p4 = sizes(2) / bfma(2);
  n4 = sizes(3) / bfma(3);

  % Use bFMA to compute matrix products.
  for ell = 1:p4
    C = single(A(:, (ell-1)*bfma(2)+1:ell*bfma(2)) *...
               B((ell-1)*bfma(2)+1:ell*bfma(2), :) + C); % C is converted to double
  end
end
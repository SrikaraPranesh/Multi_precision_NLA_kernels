function [S, errA, errB] = matmulmulticomp(A, B, bfma, maxdeg)
%   Multi component matrix-matrix multiply
  [m,p] = size(A);
  [p1,n] = size(B);

  assert(p1 == p);

  if nargin < 3
      bfma = [4,4,4];
  end
  
  if nargin < 4
    maxdeg = Inf;
  end

  [A,B,~] = bfmapadding(A,B,bfma);

  [Aexp,qA] = scaledexpansion(A);
  [Bexp,qB] = scaledexpansion(B);

  if nargout > 1
    Atmp = Aexp(:,:,qA);
    for i = qA-1:-1:1
      Atmp = 2^-11 * Atmp + Aexp(:,:,i);
    end
    errA = norm(A - Atmp) / norm(A);
  end
  if nargout > 1
    Btmp = Bexp(:,:,qB);
    for i = qB-1:-1:1
      Btmp = 2^-11 * Btmp + Bexp(:,:,i);
    end
    errB = norm(B - Btmp) / norm(B);
  end

  S = zeros(size(A,1), size(B,2));

  for t = min(qA+qB,maxdeg):-1:2
    S = 2^-(11) * S;
    for f = max(1,t - qB):min(t-1,qA)
        % the below formula doesnt seem to consider all possible
        % combinations.
      S = matmuladd(Aexp(:,:,f), Bexp(:,:,t-f), S, bfma);
    end
  end

  S = S(1:m,1:n);

end
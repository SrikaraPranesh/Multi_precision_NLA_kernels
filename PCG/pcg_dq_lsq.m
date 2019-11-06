function [x,nbe,iter,flag] = pcg_dq_lsq( A,x,b,L,tol)
%PCG_DQ_LSQ   Symmetric-preconditioned PCG in double/quad precision
%   Solves (A^{T}A)x=b by solving the preconditioned linear system
%   L^{-T}L^{-1}(A^{T}A)x=L^{-T}L^{-1}b using the Precondition Conjugate
%   Gradients ( PCG ) method.
%
%   Double precision used throughout, except in applying (U\U'\A) to a vector
%   which is done in quad precision using Advanpix multiprecision toolbox
%
%   input   A        REAL full rank matrix
%           x        REAL initial guess vector
%           b        REAL right hand side vector
%           L        REAL Cholesky factor of A^{T}A in low precision
%           tol      REAL error tolerance
%
%   output  x        REAL solution vector
%           nbe      REAL normwise backward error in 2-norm
%           iter     INTEGER number of (inner) iterations performed
%           flag     INTEGER: 0 = solution found to tolerance
%                             1 = no convergence given max_it
%
%   Note: Requires Advanpix multiprecision toolbox
%
% Reference --  Yousef Saad. Iterative Methods for Sparse Linear Systems. 
%               Second edition, Society for Industrial and Applied 
%               Mathematics, Philadelphia, PA, USA, 2003. xviii+528 pp. 
%               ISBN 0-89871-534-2. Algorithm 9.2
mp.Digits(34);



%Ensure double working precision
A = double(A);
b = double(b);
x = double(x);
L = double(L);

% Inititalization
flag = 0;
r = b-double((mp(A')*(mp(A)*mp(x))));
denom = (norm(A'*A)*norm(x))+norm(b);
z = double(mp(L)'\(mp(L)\mp(r)));
p = z;

for iter = 1:length(b)
    Ap = double(mp(A')*(mp(A)*mp(p)));
    numer = r'*z;
    alpha = numer/(p' * Ap);
    x = x + alpha * p;
    r = r - alpha * Ap;
    nbe = norm(r)/denom;
    if nbe < tol
        break;
    end
    z = double(mp(L)'\(mp(L)\mp(r)));
    p = z + (r'*z / numer) * p;
end

if (nbe >= tol)
    flag = 1;
end

end


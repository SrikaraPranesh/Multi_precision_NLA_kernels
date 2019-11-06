function [x,nbe,iter,flag] = pcg_sd_lsq(A,x,b,L,tol)
%PCG_SD_LSQ   Symmetric-preconditioned PCG in single/double precision
%   Solves (A^{T}A)x=b by solving the preconditioned linear system 
%   L^{-T}L^{-1}(A^{T}A)x=L^{-T}L^{-1}}b using the Precondition Conjugate 
%   Gradients ( PCG ) method.
%
%   Single precision used throughout, except in applying (U\U'\(A'*A)) 
%   to a vector which is done in double precision.
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
% Reference --  Yousef Saad. Iterative Methods for Sparse Linear Systems. 
%               Second edition, Society for Industrial and Applied 
%               Mathematics, Philadelphia, PA, USA, 2003. xviii+528 pp. 
%               ISBN 0-89871-534-2. Algorithm 9.2


%Ensure double working precision
A = single(A);
b = single(b);
x = single(x);
L = single(L);

% Inititalization
flag = 0;
r = b-single((double(A')*(double(A)*double(x))));
denom = (norm(A'*A)*norm(x))+norm(b);
z = single(double(L)'\(double(L)\double(r)));
p = z;

for iter = 1:length(b)
    Ap = single(double(A')*(double(A)*double(p)));
    numer = r'*z;
    alpha = numer/(p' * Ap);
    x = x + alpha * p;
    r = r - alpha * Ap;
    nbe = norm(r)/denom;
    if nbe < tol
        break;
    end
    z = single(double(L)'\(double(L)\double(r)));
    p = z + (r'*z / numer) * p;
end

if (nbe >= tol)
    flag = 1;
end

end
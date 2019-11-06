function [x,nbe,iter,flag] = pcg_sd(A,x,b,L,tol)
%PCG_SD   Symmetric-preconditioned PCG in single/double precision
%   Solves Ax=b by solving the preconditioned linear system
%   L^{-T}L^{-1}Ax=L^{-T}L^{-1}b using the Precondition Conjugate
%   Gradients ( PCG ) method.
%
%   Single precision used throughout, except in applying (U\U'\A) 
%   to a vector which is done in double precision.
%
%   input   A        REAL symmetric positive definite matrix 
%           x        REAL initial guess vector
%           b        REAL right hand side vector
%           L        REAL Cholesky factor of A in low precision
%           tol      REAL error tolerance
%
%   output  x        REAL solution vector
%           res      REAL residual norm
%           iter     INTEGER number of (inner) iterations performed
%           flag     INTEGER: 0 = solution found to tolerance
%                             1 = no convergence given max_it
%
%
% Note -- if the orignal matrix, say Ao, is directly converted into
%         single precision, then it becomes indefinite, as it might have
%         some small eigenvalues. Experimentally we have observed that
%         converting diagonally scaled matrix usually solves this problem.
%
% Reference --  Yousef Saad. Iterative Methods for Sparse Linear Systems. 
%               Second edition, Society for Industrial and Applied 
%               Mathematics, Philadelphia, PA, USA, 2003. xviii+528 pp. 
%               ISBN 0-89871-534-2. Algorithm 9.2
if (norm(A-A')~=0)
    error('Matrix is not symmetric');
% elseif (min(eig(A))<=0)
%     error('Matrix is not positive definite')
end

%Ensure double working precision
A = single(A);
b = single(b);
x = single(x);
L = single(L);

% Inititalization
flag = 0;
r = b-single(double(A)*double(x));
denom = (norm(A)*norm(x))+norm(b);
z = single(double(L)'\(double(L)\double(r)));
p = z;

for iter = 1:length(b)
    Ap = single(double(A)*double(p));
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
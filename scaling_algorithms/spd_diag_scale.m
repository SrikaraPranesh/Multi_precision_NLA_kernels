function [A,D,mu] = spd_diag_scale(A,flag,theta)
%SPD_DIAG_SCALE scales a symmetric and positive defnite matrix into 
%   the range of fp16.
%   This routine is an implementation of Algorithm 2.3 in [1] for symmteric 
%   and positive definite (SPD) matrices. Diagonal scaling is performed
%   using sqrt(diag(A)). For further details regarding diagonal scaling of 
%   SPD matrices, readers are referred to [2, p.199].
%   flag = 0, no rescaling (default value)
%   flag = 1, with rescaling
%
%   [1] Nicholas J. Higham, Srikara Pranesh, and Mawussi Zounon. 
%   "Squeezing a Matrix into Half Precision, with an Application to 
%    Solving Linear Systems." MIMS Eprint (2018).
%   [2] Nicholas J. Higham, Accuracy and stability of numerical 
%       algorithms. Vol. 80. SIAM, 2002.
if nargin < 2, flag = 0; end
if (norm(A-A')~=0)
    error('Input matrix is not symmetric')
elseif (min(diag(A)) <= 0)
    error('matrix might not be positive definite')
end

d = sqrt(diag(A));
D = diag(1./d);

% computes only on lower triangular part of the matrix
U = D*triu(A,1)*D; d1 = D*(diag(diag(A)))*D;
A = U+U'+d1;

if (flag == 1 )
    [u,xmins,xmin,xmax,p,emins,emin,emax] = float_params('h');
    rmax2 = theta * xmax;
    beta = max(max(abs(A)));
    mu = rmax2/beta;
    A = mu*A;
else
    mu = 1;
end

end


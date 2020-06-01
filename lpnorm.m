function [est, x, k] = lpnorm(A,p,flag,tol,prnt)
%LPNORM   Estimate of matrix p-norm (1 <= p <= inf). 
%        [EST, x, k] = LPNORM(A, p, TOL) estimates the Holder p-norm 
%        using the usual method if the norm of 'A' is required, and uses
%        low precision LU factors is the norm of A^{-1} is required. 
%        
%        TOL is a relative convergence tolerance (default 1E-4).
%        Returned are the norm estimate EST (which is a lower bound for the
%        exact p-norm), the corresponding approximate maximizing vector x,
%        and the number of power method iterations k.
%        A nonzero fourth input argument causes trace output to the screen.
%        If A is a vector, this routine simply returns NORM(A, p).
%        flag = 1 is used to compute the norm of A^{-1}, and flag = 0
%        is to compute the norm of A.
%
%        See also NORM, NORMEST, NORMEST1.

%        Note: The estimate is exact for p = 1, but is not always exact for
%        p = 2 or p = inf.  Code could be added to treat p = 2 and p = inf
%        separately.

%        Note: In GMRES-IR by following default parameters are used 
%               1. 2 sided diagonal scaling 
%               2. left preconditioning
%               3. maximum of 10 iterative refinement steps
%
%        Calls DUAL, float_params, chop
%
%        Reference:
%        N. J. Higham, Accuracy and Stability of Numerical Algorithms,
%           Second edition, Society for Industrial and Applied Mathematics,
%           Philadelphia, PA, 2002; sec. 15.2.
%        N. J. Higham, and S. Pranesh. Simulating Low Precision
%           Flotaing-Point Arithmetic. MIMS Eprint March 2019.

if nargin < 3, error('Must specify norm via second parameter.'), end
[m,n] = size(A);
if m ~= n, error('Input matrix must be square'), end
if min(m,n) == 1, est = norm(A,p); return, end

if nargin < 5, prnt = 0; end
if nargin < 4 | isempty(tol), tol = 1e-4; end

% parameters for GMRES-IR
if flag == 1
    % precision combinations
    if isa(A,'single')
        precw = 1; precr = 2; tol = 1e-2;
    elseif isa(A,'double')
        precw = 2; precr = 4; tol = 1e-4;
    end
    
    % scaling type
    if norm(A-A')==0
        scale.type = 's';
        if (min(eig(A))> 0)
            scale.type = 'p';
        end
    else
        scale.type = 'g';
    end
end


% Apply Algorithm PM (the power method).

q = dual(p);
k = 1;
x = ones(n,1);
x = x/norm(x,p);
est = norm((A*x),p);

scale.flag = 1; 
scale.theta = 0.1; scale.precon = 'l'; 
figs = 0; maxit = 10;


while 1
    
    if flag == 1
        % call GMRES-IR
        [y,~] = gmresir3(A,x,1,precw,precr,maxit,tol,scale,figs);
        y = double(y);
    else
        y = A*x;
    end
    est_old = est;
    est = norm(y,p);
    
    if flag == 1
        % call GMRES-IR
        [z,~] = gmresir3(A',dual(y,p),1,precw,precr,maxit,tol,...
            scale,figs);
        z = double(z);
    else
        z = A'*dual(y,p);
    end
    
    if prnt
        fprintf('%2.0f: norm(y) = %9.4e,  norm(z) = %9.4e', ...
                 k, norm(y,p), norm(z,q))
        fprintf('  rel_incr(est) = %9.4e\n', (est-est_old)/est)
    end

    if ( norm(z,q) <= z'*x | abs(est-est_old)/est <= tol ) & k > 1
       return
    end

    x = dual(z,q);
    k = k + 1;

end

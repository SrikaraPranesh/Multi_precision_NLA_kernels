function [x, error, its, flag] = lsqgmres_sd( A, x, ba, Q, R, a, restrt, max_it, tol, ptype)
%LSQGMRES_SD   Left-preconditioned GMRES in single/double precision
%   Solves the correction equation with augmented matrix using the left
%   preconditioned GMRES. The preconditioned matrix is scaled so that the
%   condition number is of the same order as that of A. The solution is
%   rescaled at the end.
%   Currently uses (preconditioned) relative residual norm to check for convergence 
%   (same as Matlab GMRES)
%   Single precision used throughout, except in applying (U\L\A) to a vector 
%   which is done in double precision
%
%   input   A        REAL nonsymmetric positive definite matrix
%           x        REAL initial guess vector
%           b        REAL right hand side vector
%           L        REAL L factor of lu(A)
%           U        REAL U factor of lu(A)
%           restrt   INTEGER number of iterations between restarts
%           max_it   INTEGER maximum number of iterations
%           tol      REAL error tolerance
%           ptype    Type of preconditioner.
%
%   output  x        REAL solution vector
%           error    REAL error norm
%           iter     INTEGER number of (inner) iterations performed
%           flag     INTEGER: 0 = solution found to tolerance
%                             1 = no convergence given max_it

flag = 0;
its = 0;

%Ensure single working precision
A = single(A);
ba = single(ba);
x = single(x);

% preconditioned and scaled rhs
b = Mb_sd(Q,R,a,ba,ptype);

Ax = MA_sd(A,Q,R,a,x,ptype);
r = b-Ax;
r = single(r);

% rtmp = b-A*x;
% r = double(L)\double(rtmp);
% r = double(U)\double(r);
% r = single(r);

bnrm2 = norm(r );
if  ( bnrm2 == 0.0 ), bnrm2 = 1.0; end


error(1) = norm( r ) / bnrm2;
if ( error(1) < tol ) return, end

[m1,n1] = size(A);                                  % initialize workspace
n = m1+n1;
m = restrt;
V(1:n,1:m+1) = single(zeros(n,m+1));
H(1:m+1,1:m) = single(zeros(m+1,m));
cs(1:m) = single(zeros(m,1));
sn(1:m) = single(zeros(m,1));
e1    = single(zeros(n,1));
e1(1) = single(1.0);

for iter = 1:max_it,                              % begin iteration
    
    Ax = MA_sd(A,Q,R,a,x,ptype);
    r = b-Ax;
    r = single(r);
    
%     rtmp = single(b-A*x);
%     r = double(U)\(double(L)\double(rtmp));
%     r = single(r);
    
    V(:,1) = r / norm( r );
    s = norm( r )*e1;
    for i = 1:m,                     % construct orthonormal basis via GS
        its = its+1;
        vcur = V(:,i);      
        
        vcur = MA_sd(A,Q,R,a,vcur,ptype);
%         vcur = double(U)\(double(L)\(double(A)*double(vcur)));
        
        w = single(vcur);
      
        for k = 1:i,
            H(k,i)= w'*V(:,k);
            w = w - H(k,i)*V(:,k);
        end
        H(i+1,i) = norm( w );
        V(:,i+1) = w / H(i+1,i);
        for k = 1:i-1,                              % apply Givens rotation
            temp     =  cs(k)*H(k,i) + sn(k)*H(k+1,i);
            H(k+1,i) = -sn(k)*H(k,i) + cs(k)*H(k+1,i);
            H(k,i)   = temp;
        end
        [cs(i),sn(i)] = rotmat( H(i,i), H(i+1,i) ); % form i-th rotation matrix
        temp   = cs(i)*s(i);                        % approximate residual norm
        s(i+1) = -sn(i)*s(i);
        s(i)   = temp;
        H(i,i) = cs(i)*H(i,i) + sn(i)*H(i+1,i);
        H(i+1,i) = 0.0;
        
        error((iter-1)*m+i+1)  = abs(s(i+1)) / bnrm2;
        if ( error((iter-1)*m+i+1) <= tol ),                        % update approximation
            y = H(1:i,1:i) \ s(1:i);                 % and exit
            addvec = V(:,1:i)*y;
            x = x + addvec;
            break;
        end
    end
    
    if ( error(end) <= tol ), break, end
    y = H(1:m,1:m) \ s(1:m);
    addvec = V(:,1:m)*y;
    x = x + addvec;                            % update approximation
    
    Ax = MA_sd(A,Q,R,a,x,ptype);
    r = b-Ax;
    r = single(r);
    
%     rtmp = b-A*x;
%     r = double(U)\(double(L)\double(rtmp));           % compute residual
%     r = single(r);
    s(i+1) = norm(r);
    error = [error,s(i+1) / bnrm2];                        % check convergence
    if ( error(end) <= tol ) 
        break; 
    end;
end

if ( error(end) > tol ) flag = 1; end;                 % converged




function [ c, s ] = rotmat( a, b )

%
% Compute the Givens rotation matrix parameters for a and b.
%
if ( b == 0.0 ),
    c = 1.0;
    s = 0.0;
elseif ( abs(b) > abs(a) ),
    temp = a / b;
    s = 1.0 / sqrt( 1.0 + temp^2 );
    c = temp * s;
else
    temp = b / a;
    c = 1.0 / sqrt( 1.0 + temp^2 );
    s = temp * c;
end


function x = MA_sd(A,Q,R,a,d,ptype)
%MA_SD Matrix-Vector product with a preconditioned augmented system in the
%   iterative refinement of the least squares problem.
%       x = MA_sd(A,Q,R,a,d)
%       A -- real mxn matrix where m>n, and full rank.
%       Q -- real mxn orthogoanl matrix.
%       R -- real nxn upper triangulat matrix.
%       d -- real (m+n)x1 input vector.
%       a -- scaling factor used in augmented system.
%       ptype -- type of preconditioner.


[m,n] = size(A);
A = double(A);
Q = double(Q); R = double(R);
a = double(a); d = double(d);
r = d((1:m),1); s = d((m+1:(m+n)),1);
tv1 = A*s; tv2 = A'*r;

if ptype == 1
    t1 = (r-(Q*(Q'*r)))+(Q*(R'\tv2));
    t2 = (tv1-(Q*(Q'*tv1)))/a;
elseif ptype == 2
    t1 = a*(r-(Q*(Q'*r)))+(Q*(R'\tv2));
    t2 = tv1-(Q*(Q'*tv1));
end

t3 = a*((R\(Q'*r))-(R\(R'\tv2)));
t4 = R\(Q'*tv1);
x = single([t1+t2;t3+t4]);


function x = Mb_sd(Q,R,a,b,ptype)
%MB_SD computes preconditioned and scaled rhs for the augmented system
%      solved using GMRES.
%      x = Mb_sd(Q,R,a,b)
%       Q -- real mxn orthogonal matrix.
%       R -- real nxn upper triangulat matrix.
%       d -- real (m+n)x1 input vector.
%       a -- scaling factor used in augmented system.
%       ptype -- type of preconditioner.

[m,n] = size(Q);
Q = double(Q); R = double(R);
a = double(a); b = double(b);
r = b((1:m),1); s = b((m+1:(m+n)),1);

if ptype == 1
    t1 = (r-(Q*(Q'*r)))/a;
elseif ptype == 2
    t1 = (r-(Q*(Q'*r)));
end

t2 = Q*(R'\s);
t3 = R\(Q'*r);
t4 = -a*(R\(R'\(s)));
x = single([t1+t2;t3+t4]);

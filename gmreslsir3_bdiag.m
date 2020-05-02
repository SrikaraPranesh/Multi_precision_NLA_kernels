function [opdata,x,iter,gmresits] = gmreslsir3_bdiag(A,b,precf,precw,precr,iter_max,gtol)
% GMRESIR3_BDIAG  GMRES-based iterative refinement in three precisions.
%     opdata = gmresir3(A,b,precf,precw,precr,iter_max,gtol)
%     solves the least-squares problem Ax = b using gmres-based
%     iterative refinement with at most iter_max ref. steps and
%     GMRES convergence tolerance gtol, with
%     QR factors computed in precision precf:
%       * half if precf = 0,
%       * single if precf = 1,
%       * double if precf = 2,
%     working precision precw:
%       * half if precw = 0
%       * single if precw = 1,
%       * double if precw = 2,
%     and residuals computed at precision precr:
%       * single if precr = 1,
%       * double if precr = 2,
%       * quad if precr = 4
%     Uses split preconditioned GMRES M1\A/M2 with 
%     block diagonal M1 and M2
% Note: Requires CleveLaboratory and Advanpix multiprecision computing toolbox


if precf ~=0 && precf ~=1 && precf ~= 2, error('precf should be 0, 1 or 2'), end
if precw ~=0 && precw ~=1 && precw ~= 2, error('precw should be 0, 1 or 2'), end
if precr ~=1 && precr ~= 2 && precr ~= 4, error('precr should be 1, 2, or 4'), end

[m,n] = size(A);
q = inf;  % p-norm to use.



if precf == 1
    fprintf('**** Factorization precision is single.\n')
elseif precf == 2
    fprintf('**** Factorization precision is double.\n')
else
    fprintf('**** Factorization precision is half.\n')
end

if precw == 0
    fprintf('**** Working precision is half.\n')
    A = fp16(A);
    b = fp16(b);
    u = eps(fp16(1));
elseif precw == 2
    fprintf('**** Working precision is double.\n')
    A = double(A);
    b = double(b);
    u = eps('double');
else
    fprintf('**** Working precision is single.\n')
    A = single(A);
    b = single(b);
    u = eps('single');
end

if precr == 1
    fprintf('**** Residual precision is single.\n')
elseif precr == 2
    fprintf('**** Residual precision is double.\n')
else
    fprintf('**** Residual precision is quad.\n')
    mp.Digits(34);
end


% Record infinity norm condition number of A
opdata.condA = double(norm(mp(A,64),'inf')*norm(pinv(mp(A,64)),'inf'));

%Compute exact solution and residual via advanpix
xact = (mp(double(A),64)\mp(double(b),64));
ract = mp(double(b),64)-mp(double(A),32)*mp(xact,64);

% Compute norms of exact solution and residual
rtn = norm(ract,2);
xtn = norm(xact,2);

% Compute QR factorization in precf
if precf == 1
    [Q,R] = qr(single(A));
    Q = single(Q);
    RR = single(R); % RR is trapezoidal factor
elseif precf == 2
    [Q,R] = qr(double(A));
    RR = R;
else
    fp.format = 'h'; chop([],fp);
    [~,~,~,xmax,~] = float_params(fp.format);
    D = diag(1./vecnorm(A));
    mu = 0.1*xmax;
    As = chop(mu*A*D);
    [Q,R] = house_qr_lp(As,0); % half precision via advanpix
    R = (1/mu)*R*diag(1./diag(D));
    RR = R(1:n, 1:n);   % RR is trapezoidal factor
end
R = RR(1:n, 1:n);   % upper triangular part of RR factor
Q1 = Q(:,1:n);

% compute the scaling factor alpha
if precw == 1
    alpha = 2^(-1/2)*(min(svd(single(R))));
elseif precw == 2
    alpha = 2^(-1/2)*(min(svd(double(R))));
end


% Compute and store initial solution and residual in working precision
if precw == 0
    x = fp16(RR\(Q'*b));
    rx = fp16(b-A*x);
elseif precw == 2
    x = double(RR\(Q'*b));
    rx = double(b-A*x);
else
    x = single(RR\(Q'*b));
    rx = single(b-A*x);
end

% Note: when kinf(A) is large, the initial solution x can have 'Inf's in it
% If so, default to using 0 as initial solution
if sum(isinf(single(x)))>0 || sum(isinf(single(rx)))>0
    x =  zeros(size(b,1),1);
    rx = b;
    fprintf('**** Warning: x0 contains Inf. Using 0 vector as initial solution.\n')
end

% Record relative error in computed initial x and r
xerr(1) = norm(double(x)-double(xact),2)./xtn;
rerr(1) = norm(double(rx)-double(ract),2)./rtn;


% Construct (scaled) augmented system
if precw == 0
    Aug_A = fp16([alpha.*eye(m), A; A', zeros(n)]);
elseif precw == 2
    Aug_A = double([alpha.*eye(m), A; A', zeros(n)]);
else
    Aug_A = single([alpha.*eye(m), A; A', zeros(n)]);
end

% Record infinity and 2-norm condition numbers of augmented system
opdata.condAugA = cond(Aug_A,'inf');
opdata.condAugA2 = cond(Aug_A,2);

% Construct block diagonal split preconditioners in precw, composed of R
% factors computed in precf
 if precw == 1
    L1 = single([eye(m), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*R']);
    R1 = single([eye(m), zeros(m,n); zeros(n,m),1/sqrt(alpha).*R]);
    L2 = eye(m+n);
    R2 = eye(m+n);
elseif precw == 2
    L1 = double([eye(m), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*R']);
    R1 = double([eye(m), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*R]);
    L2 = eye(m+n);
    R2 = eye(m+n);
 else
    % Note: fp16([alpha.*eye(m), fp16(Q1*R); fp16(R'*Q'), zeros(n)])
    % causes an error here, so I cast to single when multiplying the
    % matrices
    L1 = fp16([sqrt(alpha).*eye(m), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*single(R')]);
    R1 = fp16([sqrt(alpha).*eye(m), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*single(R)]);
    L2 = eye(m+n);
    R2 = eye(m+n);
 end


% Form "exact" preconditioned system via Advanpix
extPC = (mp(L1,64)\mp(Aug_A,64))/mp(R1,64);

% Record infinity norm condition number of exact preconditioned system
opdata.condMA = cond(double(extPC),'inf');


cged = false;
iter = 0; dx = 0; rd = 0;

% Init array to store total number of gmres iterations in each refinement step
gmresits = 0;

% Init array to store final relative (preconditioned) residual norm in gmres
gmreserr = [];


while ~cged
    
    % Increase iteration count; break if hit iter_max
    iter = iter + 1;
    if iter > iter_max, break, end
    
    % Compute residuals in precr
    if precr == 1
        f = single(b)-single(rx)-single(A)*single(x);
        g = single(-A')*single(rx);
    elseif precr == 2
        f = single(double(b)-double(rx)-double(A)*double(x));
        g = single(double(-A')*double(rx));
    else
        f = mp(mp(b,32)-mp(rx,32)-mp(A,32)*mp(x,32),32);
        g = mp(mp(-A',32)*mp(rx,32),32);
    end
    
    % Construct right-hand side for augmented scaled system
    Aug_rhs = [alpha.*f; g];
    
    %Call GMRES to solve for correction terms
    if precw == 0
        [d, err, its, ~] = gmres_hs( Aug_A, zeros(m+n,1), Aug_rhs, L1, L2, R1, R2, m+n, 1, gtol);
    elseif precw == 2
        [d, err, its, ~] = gmres_dq( Aug_A, zeros(m+n,1), Aug_rhs, L1, L2, R1, R2, m+n, 1, gtol);
    else
        [d, err, its, ~] = gmres_sd( Aug_A, zeros(m+n,1), Aug_rhs, L1, L2, R1, R2, m+n, 1, gtol);
    end
    
    
    % Pick out updates to x and r from the GMRES solution
    dr = d(1:m);
    dx = (1/alpha).*d(m+1:end);
    
    
    % Record the number of iterations gmres took
    gmresits = gmresits+its;
    
    % Record the final relative (preconditioned) residual norm in GMRES
    gmreserr = [gmreserr,err(end)];
    
    % Record relative (preconditioned) residual norm in each iteration of
    % GMRES (so we can look at convergence trajectories if need be)
    gmreserrvec{iter} = err;
    
    % Store previous solution
    xold = x;
    
    % Update solution and residual in precw
    if precw == 0
        rx = fp16(fp16(rx)+fp16(dr));
        x = fp16(fp16(x)+fp16(dx));
    elseif precw == 2
        rx = double(double(rx)+double(dr));
        x = double(double(x)+double(dx));
    else
        rx = single(single(rx)+single(dr));
        x = single(single(x)+single(dx));
    end
    
    
    % Store relative error in computed x and r
    xerr(iter+1) = mp(norm(mp(x,64)-mp(xact,64),2)./xtn,64);
    rerr(iter+1) = mp(norm(mp(rx,64)-mp(ract,64),2)./rtn,64);
    
    % Check convergence
    if(xerr(iter+1)<= u && rerr(iter+1)<=u)
        break;
    end
    
    % Compute relative change in solution
    ddx = norm(x-xold,q)/norm(x,q);
    
    % Check if ddx contains infs, nans, or is 0
    if ddx == Inf || isnan(double(ddx))
        break;
    end
    
end

if ((iter >= iter_max) && (xerr(end)>u) && (rerr(end)>u))
    iter = inf; gmresits = inf;
end

% Record vector of errors in solution and residual
opdata.xerr = xerr;
opdata.rerr = rerr;

% Record final solution and residual obtained
opdata.x = x;
opdata.x = rx;

% Record information about GMRES iterations
opdata.gmresits = gmresits;
opdata.gmreserr = gmreserr;
opdata.gmreserrvec = gmreserrvec;


end
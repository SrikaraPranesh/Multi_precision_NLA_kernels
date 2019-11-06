function [x,its,t_gmres_its] = lsq_gmresir3(A,b,precf,precw,precr,iter_max,gtol,scale)
%LSQ_GMRESIR3 Normal equation based least squares solver using GMRES-IR.
%   Solves the full rank least square problem using the normal equation
%   method. The normal equation is solved using GMRES-IR with low precision
%   Cholesky factors as preconditioners, and it also refines the solution.
%     iter_max ref. steps and GMRES convergence tolerance gtol, with
%     Cholesky factors computed in low precision precf:
%       * fp16 if precf = 1,
%       * bfloat16 if precf = 2,
%       * fp32 if precf = 3,
%     working precision precw:
%       * single if precw = 1,
%       * double if precw = 2,
%     and residuals computed at precision precr:
%       * double if precr = 2,
%       * quad if precr = 4
%       
%     
%     scale -- A structure which contains various options for
%              GMRES-IR 
%               *scale.theta - a number between (0,1].
%               *scale.pert - diagonal perturbation for low precision
%                               Cholesky factorization.
%               *scale.fact - '1' perturbs diagonaly scaled matrix
%                           - '2' perturbs the normal matrix
%   iter_max -- Maximum number of iterative refinement steps.
%
%   NOTE -- Not advisable to use this method if the matrix $A$ is
%   ill-conditioned.
%
%   NOTE -- If bfloat16 is used as the low precision format, then 
%   subnormal numbers are not supported.
%

if precf ~=1 && precf ~= 2 && precf ~= 3, error('precf should be 1, 2 or 3'), end
if precw ~=1 && precw ~= 2, error('precw should be 1 or 2'), end
if precr ~= 2 && precr ~= 4, error('precr should be 2, or 4'), end
if length(iter_max) ~= 2, error('iter_max must be a vector of size 2'), end

if precf == 1
    fprintf('**** Factorization precision is fp16.\n')
    ufs1 = 'h';
    fp.format = 'h'; fp.round = 1;
    chop([],fp);
elseif precf == 2
    fprintf('**** Factorization precision is bfloat16.\n')
    ufs1 = 'b';
    fp.format = 'b'; fp.round = 1;
    chop([],fp);
elseif precf == 3
    fprintf('**** Factorization precision is fp32.\n')
    ufs1 = 's';
    fp.format = 's'; fp.round = 1;
    chop([],fp);
end

if precw == 2
    fprintf('**** Working precision is double.\n')
%     uws = 'double';
    A = double(A);
    b = double(b);
    u = eps(double(1/2));
else
    fprintf('**** Working precision is single.\n')
%     uws = 'single';
    A = single(A);
    b = single(b);
    u = eps(single(1/2));
end

if precr == 2
    fprintf('**** Residual precision is double.\n')
%     urs = 'double';
else
    fprintf('**** Residual precision is quad.\n')
%     urs = 'quad';
    mp.Digits(34);
end

[~,n] = size(A);
if (ufs1 == 'h') 
    S = diag(1./vecnorm(A));
elseif (ufs1 == 'b') || (ufs1 == 's')
    S = eye(n,n);
end

if (precf == 1)
    [uh,~,~,xmax] = float_params('h');
    mu = scale.theta*xmax;
elseif (precf == 2)
    [uh,~] = float_params('b');
    mu = 1;
elseif (precf == 3)
    [uh,~] = float_params('s');
    mu = 1;
else
    error('unknown half precision format')
end



if (ufs1 == 'h') || (ufs1 == 'b')
    Bh = chop(sqrt(mu)*A*S);
    % construction of the normal equation
    C = hgemm(Bh',Bh,ufs1);
elseif ufs1 == 's'
    Bh = single(sqrt(mu)*A*S);
    % construction of the normal equation
    C = Bh'*Bh;
end


% perturbation 
if ufs1 == 'h'
    E = mu*scale.pert*uh*eye(length(C));
elseif ufs1 == 'b' || ufs1 == 's'
    E = mu*scale.pert*uh*diag(diag(C));
end
B1 = C+E;
if (ufs1 == 'h') || (ufs1 == 'b')
    B1 = chop(B1,fp);
    [R,flag] = chol_lp(B1,ufs1);
    if flag
        error('Cholesky factorization was not successful')
    end
elseif ufs1 == 's'
    R = chol(B1);
end
LL = (1/mu)*diag(1./diag(S))*double(R');
U = double(R)*diag(1./diag(S));



bh = lp_matvec(S',lp_matvec(A',b,ufs1),ufs1);
% Initial solution 
y = trisol(R,trisol(R',bh,ufs1),ufs1);
x = mu*S*y;

% x = zeros(length(x),1);

% Start of Iterative refinement
t_gmres_its = 0;
for its = 1:iter_max(1,1)
    
    if norm(x)~= 0
        nbe = lsq_be(A,x,b,1);
    else
        nbe = 1;
    end
    
    if nbe < length(b)*u
        its = its-1;
        break
    end
     
    %Compute residual vector
    if precr == 2
        r = double(A')*(double(b) - double(A)*double(x));
    else
        r = double(mp(A'))*(mp(double(b),34) - mp(double(A),34)*mp(double(x),34));
    end
    
    scale.precon = 'l';
    %Call GMRES to solve for correction term
    if scale.precon == 'l'
        if (precw == 2 && precr == 4) 
            [d, err, iter, con_flag] = gmres_dq_lsq(A, zeros(n,1), double(r), LL, U, n, 1, gtol);
        elseif (precw == 1 && precr == 2) 
            [d, err, iter, con_flag] = gmres_sd_lsq(A, single(zeros(n,1)), single(r), LL, U, n, 1, gtol);
        elseif (precw == 2 && precr == 2) 
            [d, err, iter, con_flag] = gmres_dd_lsq(A, double(zeros(n,1)), double(r), LL, U, n, 1, gtol);
        end
    elseif scale.precon == 's'
        if precw == 2
            [d, err, iter, con_flag] = gmres_dq_2((A'*A), zeros(n,1), double(r), LL, U, n, gtol);
        else
            [d, err, iter, con_flag] = gmres_sd_2((A'*A), single(zeros(n,1)), single(r), LL, U, n, gtol);
        end
    end

    t_gmres_its = t_gmres_its+iter;

                                                                      
    %Update solution
    if precw == 2
        x = x + double(d);
    else
        x = x + single(d);
    end
end

if norm(x)~=0
    nbe = lsq_be(A,x,b,1);
else
    nbe = 1;
end

if (its == iter_max(1,1)) && (nbe > length(b)*u)
    its = Inf;
end

%%% Backward error of a least squares problem
%%%%%%%%%%%%%%%%%%%%%%%
function eta = lsq_be(A,x,b,xi)
% for more details refer to 'Accuracy and Stability of Numerical
% Algorithms -- Nicholas J. Higham', Second Edition, p -- 393.
r1 = double(mp(b)-(mp(A)*mp(x)));
[m,~] = size(A);
t = (xi^2)*(x'*x);
if xi ~= Inf
    mu1 = t/1+t;
else
    mu1 = 1;
end
phi = sqrt(mu1)*(norm(r1)/norm(x));
D = [A (phi*(eye(m)-(r1*pinv(r1))))];
smin = min(svd(D));
eta = min([phi smin])/norm([A b],'fro');
end


end


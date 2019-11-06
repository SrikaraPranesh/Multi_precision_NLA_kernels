function [x,iter_o,gmres_its_o] = gmresir3(A,b,precf,precw,precr,iter_max,...
                                       gtol,scale,figs)
%GMRESIR3  GMRES-based iterative refinement in three precisions.
%     x = gmresir3(A,b,precf,precw,precr,iter_max,gtol,scale) solves
%     Ax = b using gmres-based iterative refinement with at most
%     iter_max ref. steps and GMRES convergence tolerance gtol, with
%     LU factors computed in low precision precf:
%       * fp16 if precf = 1,
%       * bfloat16 if precf = 2,
%       * fp32 if precf = 3;
%     working precision precw:
%       * single if precw = 1,
%       * double if precw = 2,
%     and residuals computed at precision precr:
%       * double if precr = 2,
%       * quad if precr = 4
%       
%     figs = 1 will plot the forward, backward errors 
%            and bounds from the analysis of [1].
%     
%     scale -- A structure which contains various options for
%              squeezing a matrix into low precision range. 
%              Implementation based on [2].
%               *scale.flag = 1 to call 2 sided diagonal scaling
%                                                      
%               *scale.cluf = 1 custom lu factorization
%
%               *scale.type = 'g','s','p'('general','symmetric',
%                                          'positive').
%               *scale.theta - a number between (0,1].
%               *scale.pert - diagonal perturbation for low precision
%                               Cholesky factorization.
%               *scale.precon = 'l' or 's' ('left', 'symmetric'.)
%
%               *scale.cri = 1 uses normwise, componentwise backward error
%               and forward error as a stopping criterion. = 2 just
%               normwise backward error <= nu. Default value is 2.
%   Note: Requires chop.m and Advanpix multiprecision toolbox
%
%
%
%   Note: 
% i. if iter_o,gmres_its_o is '10^4' then it means solution of the
%update equation has NaN or Inf.
% ii. if iter_o, gmres_its_o is '-1e4' then it means the computed low
% precision LU factors are singular.

if ~isfield(scale,'cri'), scale.cri = 2; end
if ~isfield(scale,'cluf'), scale.cluf = 0; end

if precf ~=1 && precf ~= 2 && precf ~= 3, error('precf should be 1 or 3'), end
if precw ~=1 && precw ~= 2, error('precw should be 1 or 2'), end
if precr ~= 2 && precr ~= 4, error('precr should be 2, or 4'), end

n = length(A);

if precf == 1
    fprintf('**** Factorization precision is fp16.\n')
    ufs = 'half';
    ufs1 = 'h';
elseif precf == 2
    fprintf('**** Factorization precision is bfloat16.\n')
    ufs = 'bfloat16';
    ufs1 = 'b';
else
    fprintf('**** Factorization precision is fp32.\n')
    ufs = 'single';
    ufs1 = 's';
end

if precw == 2
    fprintf('**** Working precision is double.\n')
    uws = 'double';
    A = double(A);
    b = double(b);
    u = eps(double(1/2));
else
    fprintf('**** Working precision is single.\n')
    uws = 'single';
    A = single(A);
    b = single(b);
    u = eps(single(1/2));
end

if precr == 2
    fprintf('**** Residual precision is double.\n')
    urs = 'double';
else
    fprintf('**** Residual precision is quad.\n')
    urs = 'quad';
    mp.Digits(34);
end

xact = double(mp(double(A),34)\mp(double(b),34));

%Compute low precision factorization
[uh,xmins,xmin,xmax] = float_params(ufs);
if (ufs1 == 'h' && scale.flag == 1)
    if scale.type == 'g'
        [Ah,R,C] = scale_diag_2side(A);
    elseif scale.type == 's'
        [Ah,R,C] = scale_diag_2side_symm(A);
    elseif scale.type == 'p'
        [Ah,R] = spd_diag_scale(A);
        C = R;
        c = scale.pert;
        Ah = Ah+(c*uh*eye(n));
    else
        error('unknown matrix type');
    end
    mu = (scale.theta)*xmax;
    Ah = mu*Ah;
    if scale.type == 'p'
        U = chol_lp(Ah);
        L = U'; P = eye(n);
    else
        Ah = chop(Ah);
        [L,U,p] = lutx_chop(Ah);
        I = chop(eye(n)); P = I(p,:);
    end
    LL = (double(P')*double(L));
    LL = (1/mu)*diag(1./diag(R))*double(LL);
    U = double(U)*diag(1./diag(C));
    x = U\(LL\b);
elseif ufs1 == 's'
    c = scale.pert;
    Ah = A+(c*uh*diag(diag(A)));
    U = chol(single(Ah));
    L = U'; P = eye(n);
    LL = (double(P')*double(L));
    x =  U\(L\(P*b));
elseif ufs1 == 'b'
    if scale.type == 'p'
        c = scale.pert;
        A1 = A+(c*uh*diag(diag(A)));
        U = chol_lp(A1,ufs1);
        L = U'; P = eye(n);
    else
        Ah = chop(A);
        [L,U,p] = lutx_chop(Ah);
        I = chop(eye(n)); P = I(p,:);
    end
    LL = P'*L;
    x =  U\(L\(P*b));
elseif scale.cluf == 1
    L = scale.L;
    U = scale.U;
    P = scale.P;
    LL = P'*L;
    x =  U\(L\(P*b));
end


%Compute condition number of A, of preconditioned system At, cond(A), and
%cond(A,x) for the exact solution
At = double(mp(double(U),34)\(mp(double(LL),34)\(mp(double(A),34))));
kinfA = double(cond(mp(double(A),34),'inf'));
kinfAt = double(cond(mp(double(At),34),'inf'));
condAx = norm(abs(inv(mp(double(A),34)))*abs(mp(double(A),34))*abs(xact),inf)/norm(xact,inf);
condA = norm(abs(inv(mp(double(A),34)))*abs(mp(double(A),34)),'inf');

%Note: when kinf(A) is large, the initial solution x can have 'Inf's in it
%If so, default to using 0 as initial solution
if sum(isinf(single(x)))>0
    x =  zeros(size(b,1),1);
    fprintf('**** Warning: x0 contains Inf. Using 0 vector as initial solution.\n')
end

%Store initial solution in working precision
if precw == 2
    x = double(x);
else
    x = single(x);
end

cged = false;
iter = 0; dx = 0; rd = 0;

%Array to store total number of gmres iterations in each ref step
gmresits = [];

%Array to store final relative (preconditioned) residual norm in gmres
gmreserr = [];
gmres_its = 0;
while ~cged
    
    %Compute size of errors, quantities in bounds
    ferr(iter+1) = double(norm(mp(double(x),34)-mp(xact,34),'inf')/norm(mp(xact,34),'inf'));
    mu(iter+1) = norm(double(A)*(mp(double(x),34)-mp(xact,34)),'inf')/(norm(mp(double(A),34),'inf')*norm(mp(double(x),34)-mp(xact,34),'inf'));
    res = double(b) - double(A)*double(x);
    nbe(iter+1) = double(norm(mp(res,34),'inf')/(norm(mp(double(A),34),'inf')*norm(mp(double(x),34),'inf')+ norm(mp(double(b),34),'inf')));
    temp = double( abs(mp(res,34)) ./ (abs(mp(double(A),34))*abs(mp(double(x),34)) + abs(mp(double(b),34))) );
    temp(isnan(temp)) = 0; % Set 0/0 to 0.
    cbe(iter+1) = max(temp);
    
    iter = iter + 1;
    if iter > iter_max 
        gmres_its_o = inf; iter_o = inf;
        break; 
    end
    
    if isnan(kinfAt)
        gmres_its_o = -1e4; iter_o = -1e4;
        break;
    end
    
    %Check convergence
    if scale.cri == 1
        if max([ferr(iter) nbe(iter) cbe(iter)]) <= u
            gmres_its_o = gmres_its; iter_o = iter;
            break; 
        end
    elseif scale.cri == 2
        if nbe(iter) <= (length(b)*u) 
            gmres_its_o = gmres_its; iter_o = iter;
            break 
        end
    end
    
    %Compute residual vector
    if precr == 2
        rd = double(b) - double(A)*double(x);
    else
        rd = mp(double(b),34) - mp(double(A),34)*mp(double(x),34);
    end
    
    %Scale residual vector
    norm_rd = norm(rd,inf);
    rd1 = rd/norm_rd;
    
    %Call GMRES to solve for correction term
    if scale.precon == 'l'
        if (precw == 2 && precr == 4)
            [d, err, its, con_flag] = gmres_dq( A, zeros(n,1), double(rd1), LL, U, n, 1, gtol);
        elseif (precw == 1 && precr == 2)
            [d, err, its, con_flag] = gmres_sd( A, single(zeros(n,1)), single(rd1), LL, U, n, 1, gtol);
        elseif (precw == 2 && precr == 2)
            [d, err, its, con_flag] = gmres_dd( A, double(zeros(n,1)), double(rd1), LL, U, n, 1, gtol);
        end
    elseif scale.precon == 's'
        if precw == 2
            [d, err, its, con_flag] = gmres_dq_2( A, zeros(n,1), double(rd1), LL, U, n, gtol);
        else
            [d, err, its, con_flag] = gmres_sd_2( A, single(zeros(n,1)), single(rd1), LL, U, n, gtol);
        end
    end

    gmres_its = gmres_its+its;
    %Compute quantities in bounds for plotting
    lim(iter) = double( 2*u*cond(mp(double(A),34),'inf')*mu(iter));
    lim2(iter) = double(2*u*condA);
    dact = mp(double(A),34)\mp(double(rd1),34);
    etai(iter) = norm(double(mp(double(d),34)-dact),'inf')/norm(dact,'inf');
    phi(iter) = min(lim(iter),lim2(iter))+etai(iter);
    
    %Record number of iterations gmres took
    gmresits = [gmresits,its];
    
    %Record final relative (preconditioned) residual norm in GMRES
    gmreserr = [gmreserr,err(end)];
    
    %Record relative (preconditioned) residual norm in each iteration of
    %GMRES (so we can look at convergence trajectories if need be)
    gmreserrvec{iter} = err;
    
    xold = x;
    
    %Update solution
    if precw == 2
        x = x + norm_rd*double(d);
    else
        x = x + single(norm_rd)*single(d);
    end
    dx = norm(x-xold,'inf')/norm(x,'inf');
    
    %Check if dx contains infs, nans, or is 0
    if dx == Inf || isnan(double(dx))
        plt = 0;
        gmres_its_o = 1e4; iter_o = 1e4;
        break;
    end
    
end


%Generate plots
if (figs == 1 && iter > 2)
    %Create ferr, nbe, cbe plot
    fig1 = figure();
    semilogy(0:iter-1, ferr,'-rx', 'MarkerSize',20);
    hold on
    semilogy(0:iter-1, nbe,'-bo', 'MarkerSize',20);
    hold on
    semilogy(0:iter-1, cbe,'-gv', 'MarkerSize',20);
    hold on
    semilogy(0:iter-1, double(u)*ones(iter,1), '--k');
    
    %Ensure only integers labeled on x axis
    atm = get(gca,'xticklabels');
    m = str2double(atm);
    xlab = [];
    num = 1;
    for i = 1:numel(m)
        if ceil(m(i)) == m(i)
            xlab(num) = m(i);
            num = num + 1;
        end
    end
    set(gca,'xticklabels',xlab);
    set(gca,'xtick',xlab);
    xlabel({'refinement step'},'Interpreter','latex');
    set(gca,'FontSize',35)
    str_e = sprintf('%0.1e',kinfA);
    tt = strcat('$$\, \kappa_{\infty}(A) = ',str_e,', \, (u_f,u,u_r) = $$ (',ufs,',',uws,',',urs,')');
    title(tt,'Interpreter','latex');
    
    [h,objh] = legend('ferr','nbe','cbe');
    set(h,'Interpreter','latex');
    objhl = findobj(objh, 'type', 'patch'); %// objects of legend of type line
    set(objhl, 'Markersize', 35); %// set marker size as desired
    
    %%%Create phi plot
    fig2 = figure();
    semilogy(0:iter-2, lim, '-cx', 'MarkerSize',20);
    hold on
    semilogy(0:iter-2, lim2, '-+','Color',[1 0.600000023841858 0.200000002980232], 'MarkerSize',20);
    hold on
    semilogy(0:iter-2, etai, '-mo', 'MarkerSize',20);
    hold on
    semilogy(0:iter-2, phi, '-kv', 'MarkerSize',20);
    hold on
    semilogy(0:iter-1, ones(iter,1), '--k', 'MarkerSize',20);
    set(gca,'FontSize',35)
    %%%Use same x labels as error plot
    set(gca,'xticklabels',xlab);
    set(gca,'xtick',xlab);
    xlabel({'refinement step'},'Interpreter','latex');
    
    title(tt,'Interpreter','latex');
    
    h = legend('$2u_s \kappa_{\infty}(A)\mu_i$','$2u_s$cond$(A)$', '$u_s \Vert E_i \Vert_\infty$','$\phi_i$');
    set(h,'Interpreter','latex');
    objhl = findobj(objh, 'type', 'patch'); %// objects of legend of type line
    set(objhl, 'Markersize', 40); %// set marker size as desired
    
    % relative error of correction equations
    fig3 = figure();
    semilogy(0:iter-2, etai, '-mo', 'MarkerSize',20);
    
    %Use same x labels as error plot
    set(gca,'xticklabels',xlab);
    set(gca,'xtick',xlab);
    xlabel({'refinement step'},'Interpreter','latex');
    ylabel({'$\eta$'},'Interpreter','latex');
    title(tt,'Interpreter','latex');
    set(gca,'FontSize',35)
end
end

function [x,iter1] = sir3(A,b,precf,precw,precr,iter_max,figs,L,U,P)
%SIR3   LU-based iterative refinement in three precisions.
%     x = sir3(A,b,precf,precw,precr, iter_max) solves Ax = b using LU-based
%     iterative refinement (with at most iter_max ref. steps), with
%     LU factors computed in precision precf:
%       * half if precf = 0,
%       * single if precf = 1,
%       * double if precf = 2,
%     working precision precw:
%       * half if precw = 0,
%       * single if precw = 1,
%       * double if precw = 2,
%     and residuals computed at precision precr:
%       * single if precr = 1,
%       * double if precr = 2,
%       * quad if precr = 4
%
% Note: requires Cleve Laboratory and Advanpix multiprecision toolbox

if precf ~=0 && precf ~=1 && precf ~= 2, error('precf should be 0, 1 or 2'), end
if precw ~=0 && precw ~=1 && precw ~= 2, error('precw should be 0, 1 or 2'), end
if precr ~=1 && precr ~= 2 && precr ~= 4, error('precr should be 1, 2, or 4'), end

if nargin < 8
    L = []; U = [];
end

%
n = length(A);

if precf == 1
    fprintf('**** Factorization precision is single.\n')
    ufs = 'single';
elseif precf == 2
    fprintf('**** Factorization precision is double.\n')
    ufs = 'double';
else
    fprintf('**** Factorization precision is half.\n')
    ufs = 'half';
    ufs1 = 'h';
    fp.format = 'h';
    chop([],fp);
end

if precw == 0
    fprintf('**** Working precision is half.\n')
    uws = 'half';
    A = chop(A);
    b = chop(b);
    u = float_params('h');
elseif precw == 2
    fprintf('**** Working precision is double.\n')
    uws = 'double';
    A = double(A);
    b = double(b);
    u = eps('double');
else
    fprintf('**** Working precision is single.\n')
    uws = 'single';
    A = single(A);
    b = single(b);
    u = eps('single');
end

if precr == 1
    fprintf('**** Residual precision is single.\n')
    urs = 'single';
elseif precr == 2
    fprintf('**** Residual precision is double.\n')
    urs = 'double';
else
    fprintf('**** Residual precision is quad.\n')
    urs = 'quad';
    mp.Digits(34);
end

if figs == 1
    xact = double(mp(double(A),34)\mp(double(b),34));  
end

%Compute LU factorization
if isempty(L) || isempty(U) || isempty(P)
    if precf == 1
        [L,U,P] = lu(single(A));
        x =  U\(L\(P*single(b)) );
    elseif precf == 2
        [L,U,P] = lu(double(A));
        x =  U\(L\(P*double(b)) );
    else
        A = chop(A);
        [L,U,p] = lutx_chop(A);
        I = chop(eye(n)); P = I(p,:);
        t1 = lp_matvec(P,chop(b));
        t1 = trisol(L,t1);
        x = trisol(U,t1);
    end
else
    if precf == 1
        x =  U\(L\(P*single(b)));
    elseif precf == 2
        x =  U\(L\(P*double(b)));
    else
        t1 = lp_matvec(P,chop(b));
        t1 = trisol(L,t1);
        x = trisol(U,t1);
    end
end

%Compute condition number of A, cond(A), and
%cond(A,x) for the exact solution
if figs == 1
    kinfA = cond(mp(double(A),34),'inf');
    condAx = norm(abs(inv(mp(double(A),34)))*abs(mp(double(A),34))*abs(xact),inf)/norm(xact,inf);
    condA = norm(abs(inv(mp(double(A),34)))*abs(mp(double(A),34)),'inf');
end


%Note: when kinf(A) is large, the initial solution x can have 'Inf's in it
%If so, default to using 0 as initial solution
if sum(isinf(single(x)))>0
    x =  zeros(size(b,1),1);
    fprintf('**** Warning: x0 contains Inf. Using 0 vector as initial solution.\n')
end

%Store initial solution in working precision
if precw == 0
    x = chop(x);
elseif precw == 2
    x = double(x);
else
    x = single(x);
end


cged = false;
iter = 0; dx = 0; rd = 0; flag = 0;

while ~cged
    
    %Compute size of errors, quantities in bound   
    res = double(b) - double(A)*double(x);
    nbe(iter+1) = double(norm(mp(res,34),'inf')/(norm(mp(double(A),34),'inf')*norm(mp(double(x),34),'inf')+ norm(mp(double(b),34),'inf')));

    if figs == 1
        ferr(iter+1) = double(norm(mp(double(x),34)-mp(xact,34),'inf')/norm(mp(xact,34),'inf'));
        mu(iter+1) = norm(double(A)*(mp(double(x),34)-mp(xact,34)),'inf')/(norm(mp(double(A),34),'inf')*norm(mp(double(x),34)-mp(xact,34),'inf')); 
        temp = double( abs(mp(res,34)) ./ (abs(mp(double(A),34))*abs(mp(double(x),34)) + abs(mp(double(b),34))) );
        temp(isnan(temp)) = 0; % Set 0/0 to 0.
        cbe(iter+1) = max(temp);
    end
    
    iter = iter + 1;
    if iter > iter_max 
        flag = 1;
        break;
    end
    
    %Check convergence
    % if max([ferr(iter) nbe(iter) cbe(iter)]) <= u, break, end
    if nbe(iter) <= (length(b)*u), break, end
        
    %Compute residual vector
    if precr == 1
        rd = single(b) - single(A)*single(x);
    elseif precr == 2
        rd = double(b) - double(A)*double(x);
    else
        rd = double(mp(double(b),34) - mp(double(A),34)*mp(double(x),34));
    end
    
    %Scale residual vector
    norm_rd = norm(rd,inf);
    rd1 = rd/norm_rd;
    
    %Solve for correction term
    if precw == 1
        d =  U\(L\(P*single(rd1)) );
    elseif precw == 2
        d =  U\(L\(P*double(rd1)) );
    else
        t1 = lp_matvec(P,rd1);
        t1 = trisol(L,t1);
        d = trisol(U,t1);
        % d = U\(L\(P*fp16(rd1)));
    end
    
    %Compute quantities in bounds for plotting
    if figs == 1
        lim(iter) = double( 2*u*cond(mp(double(A),34),'inf')*mu(iter));
        lim2(iter) = double(2*u*condA);
        dact = mp(double(A),34)\mp(double(rd1),34);
        etai(iter) = norm(double(mp(double(d),34)-dact),'inf')/norm(dact,'inf');
        phi(iter) = min(lim(iter),lim2(iter))+etai(iter);
    end
    
    
    xold = x;
    
    %Update solution
    if precw == 0
        x = x + chop(norm_rd)*chop(d);
    elseif precw == 2
        x = x + norm_rd*double(d);
    else
        x = x + single(norm_rd)*single(d);
    end
    dx = norm(x-xold,'inf')/norm(x,'inf');
    
    %Check if dx contains infs, nans, or is 0
    if dx == Inf || isnan(double(dx))
        break;
    end
    
end


if flag == 0
    iter1 = iter-1;
else
    iter1 = Inf;
end


%Generate plots
if figs == 1
    %plot ferr, nbe, cbe
    fig1 = figure();
    semilogy(0:iter-1, ferr, '-rx');
    hold on
    semilogy(0:iter-1, nbe, '-bo');
    hold on
    semilogy(0:iter-1, cbe, '-gv');
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
    
    str_e = sprintf('%0.1e',kinfA);
    tt = strcat('LU-IR,  $$\, \kappa_{\infty}(A) = ',str_e,', \, (u_f,u,u_r) = $$ (',ufs,',',uws,',',urs,')');
    title(tt,'Interpreter','latex');
    
    h = legend('ferr','nbe','cbe');
    set(h,'Interpreter','latex');
    
    
    %Create phi plot
    fig2 = figure();
    semilogy(0:iter-2, lim, '-cx');
    hold on
    semilogy(0:iter-2, lim2, '-+','Color',[1 0.600000023841858 0.200000002980232]);
    hold on
    semilogy(0:iter-2, etai, '-mo');
    hold on
    semilogy(0:iter-2, phi, '-kv');
    hold on
    semilogy(0:iter-1, ones(iter,1), '--k');
    
    %Use same x labels as error plot
    set(gca,'xticklabels',xlab);
    set(gca,'xtick',xlab);
    xlabel({'refinement step'},'Interpreter','latex');
    
    title(tt,'Interpreter','latex');
    
    h = legend('$2u_s \kappa_{\infty}(A)\mu_i$','$2u_s$cond$(A)$', '$u_s \Vert E_i \Vert_\infty$','$\phi_i$');
    set(h,'Interpreter','latex');
    
end

end

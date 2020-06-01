%WORK1 tester for various multiprecision kernels
clear all; close all;
rng(1)
m = 10; n = 4;
A = gallery('randsvd',[m,n],1e7);
[Qh,Rh] = gs_m_chop(A,'h');
[Qm,Rm] = qr_multi_prec(A,2);
Is = Qm'*Qm; As = Qm*Rm;
% [Rh,Qh] = house_qr_chop(A,'h'); Ih = Qh'*Qh/2;
% Ah = Qh*Rh;
% norm(A-Ah)/norm(A)
norm(eye(n)-Is,inf)

% fp.format = 'h';
% a = rand(n,1); b = rand(n,1);
% xd = a'*b;
% xs = single(a')*single(b);
% xh = ip_chop_compensate(a,b);
% 
% abs(xd-xs)/abs(xd)
% abs(xd-xh)/abs(xd)

% [Rh,Uh] = house_qr_chop(A,fp);
% [Ud,Rd] = qr(single(A));
% Ih = (Uh'*Uh)/2;
% Id = Ud(:,(1:n))'*Ud(:,(1:n));
% 
% errd = norm(Id-eye(n))
% errh = norm(Ih-eye(n))

% n = 100;
% maxit = 10;
% kappa = 1e7;
% 
% rng(1);
% A = gallery('randsvd',n,-kappa);
% b = A*ones(n,1);
% [u,xmins,xmin,xmax] = float_params('h');
% tol = 1e-5; max_it = n;
% U = chop(chol(chop(A)+(u*eye(n)))); 
% L = U';
% % U = eye(n); L = U;
% x = zeros(n,1);
% 
% [x, error, its, flag] = gmres_sd_2( A, x, b, L, U, max_it, tol);

% norm(x-ones(n,1),inf)
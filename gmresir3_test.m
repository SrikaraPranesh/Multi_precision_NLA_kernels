%IR3_EXAMPLE    Example script for running iterative refinement with 3 precisions
%   Generates data for plots in Figure 10.1 of MIMS Eprint 2017.# 
%   Note: Requires Advanpix multiprecision toolbox
clear all; close all;
n = 100;
maxit = 10;
kappa = 1e12;
mp.Digits(34);
rng(1);
A = gallery('randsvd',n,kappa);
% b = double(mp(A)*mp(ones(n,1)));
b = randn(n,1);
%Run GMRES-IR with uf = half, u = double, ur = quad

% general matrix
scale.flag = 0; scale.type = 'g';
scale.theta = 0.1; scale.precon = 'l'; 
scale.cri = 1; scale.luf = 1;
figs = 1;
[x,iter,gmres_its] = gmresir3(A,b,1,2,4,maxit,1e-4,scale,figs);

% % symmetric matrix
% scale.flag = 1; scale.type = 's';
% scale.theta = 0.1;
% A = A+A'/2;
% [x,iter,gmres_its] = gmresir3(A,b,0,2,4,maxit,1e-4,scale);
% 
% % posdef matrix
% scale.flag = 1; scale.type = 'p';
% scale.theta = 0.1;
% A = gallery('randsvd',n,-kappa);
% [x,iter,gmres_its] = gmresir3(A,b,0,2,4,maxit,1e-4,scale);



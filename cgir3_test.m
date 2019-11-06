% CGIR3_TEST This function tests cgir3.m function

clear all;
close all;

n = 100;
maxit = 10;
kappa = 1e3;

rng(1);
A = gallery('randsvd',n,-kappa);
b = randn(n,1);

%Run GMRES-IR with uf = half, u = double, ur = quad

% general matrix
scale.flag = 0; scale.theta = 0.1;
scale.pert = 1; figs = 1;
[x,iter,gmres_its] = cgir3(A,b,1,1,2,maxit,1e-2,scale,figs);
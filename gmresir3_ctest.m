% GMRESIR3_CTEST  tester for matrices from SuiteSpare collection, using
%   GMRES-IR with more accurate LU factors.
clear all; close all;
rng(1);
fp.format = 'h';
[u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(fp.format);

index = ssget;
indlist = find(index.isReal == 1 & index.nrows >= 100 & ...
    index.nrows <= 500 & index.nrows == index.ncols & index.xmax >= xmax);
[nlist,i] = sort(index.nrows(indlist)) ;
indlist   = indlist(i);
nn = length(indlist);



nn = 19;



nlist = nlist';
mat_prop(:,1) = nlist;
ctest = zeros(nn,1);

fid1 = fopen('gmresir3_ctest.txt','w');

% GMRES-IR parameters
scale.flag = 1; scale.type = 'g';
scale.theta = 0.1; scale.precon = 'l';
scale.cri = 2;
figs = 0; maxit = 10;
iter = zeros(nn,2); gmres_its = zeros(nn,2);
req_qty = zeros(nn,4);

for j= 1:nn
    
    Problem = ssget(indlist(j));
    A = full(Problem.A);
    n = length(A);
    b = randn(n,1);
    
    % first perform diagonal scaling and then perform LU
    [Ah,R,C] = scale_diag_2side_symm(A);
    mu = (scale.theta)*xmax;
    Ah = mu*Ah;
    
    % residual in LU factorization and condition number of the
    % left preconditioned matrix.
    
    % half precision
    Ah1 = chop(Ah);
    [Lh,Uh,p] = lutx_chop(Ah1);
    I = chop(eye(n)); Ph = I(p,:);
    Lh = (double(Ph')*double(Lh));
    Lh = (1/mu)*diag(1./diag(R))*double(Lh);
    Uh = double(Uh)*diag(1./diag(C));
    req_qty(j,1) = norm(A-(Lh*Uh),inf)/norm(A,inf);
    req_qty(j,2) = cond(Uh\(Lh\A),inf);
    
    % compensated GEMM
    [Lc,Uc,Pc] = cgemm_lu(Ah);
    Lc = (double(Pc')*double(Lc));
    Lc = (1/mu)*diag(1./diag(R))*double(Lc);
    Uc = double(Uc)*diag(1./diag(C));
    req_qty(j,3) = norm(A-(Lc*Uc),inf)/norm(A,inf);
    req_qty(j,4) = cond(Uc\(Lc\A),inf);
    
    
    AbsA = abs(A);
    mat_prop(j,3) = max(max(AbsA));
    mat_prop(j,4) = min(AbsA(AbsA>0));
    mat_prop(j,2) = cond(A,inf);
    
    for i = 1:2
        scale.luf = i;
        fprintf('Processing matrix %d || test %d || size %d || Total matrices %d\n',j,i,n,nn);
        [x,iter(j,i),gmres_its(j,i)] = gmresir3(A,b,1,2,4,maxit,1e-4,scale,figs);
        iter(j,i) = iter(j,i)-1;
    end
    
end


% print matrix properties.
for j=1:nn
    mi = indlist(j);
    fprintf(fid1,'%d & %s & %d & %6.2e & %6.2e & %6.2e\\\\\n',...
        j,index.Name{mi,1},mat_prop(j,1),mat_prop(j,2),...
        mat_prop(j,3),mat_prop(j,4));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');

% number of ir steps and GMRES iterations.
for j=1:nn
    fprintf(fid1,'%d & %d & (%d) & %d & (%d)\\\\\n',...
        j,gmres_its(j,1),iter(j,1),gmres_its(j,2),...
        iter(j,2));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');

%residual and condition number.
for j=1:nn
    fprintf(fid1,'%d & %6.2e & %6.2e & %6.2e & %6.2e\\\\\n',...
        j,req_qty(j,1),req_qty(j,2),req_qty(j,3),...
        req_qty(j,4));
end

fclose(fid1);






function [Q,R] = house_qr_lp(A,flag,format)
% Householder reflections for QR decomposition.
% [R,U] = house_qr(A) returns
% R, the upper triangular factor, and
% U, the reflector generators for use by house_apply.
% A -- m X n real input matrix
% format -- desired low precision format, by default fp16 is used
% flag -- 0 to use the economy mode, and 1 by default

if nargin<2, format = 'h'; flag = 1; end
if nargin<3, format = 'h'; end
fp.format = format; chop([],fp);

H = @(u,x) chop(x - chop(u*(hgemm(u',x))));
[m,n] = size(A);
n1 = n; m1 = m;
% if flag == 1
%     if m>n
%         A1 = randn(m,(m-n));
%         A = [A A1];
%         n = m;
    if m<n
        error('Matrix must have more rows than columns');
    end
% end

U = zeros(m,n);
R = A;
for j = 1:min(m,n)
    u = house_gen(R(j:m,j));
    U(j:m,j) = u;
    R(j:m,j:n) = H(u,R(j:m,j:n));
    R(j+1:m,j) = 0;
end
if flag == 0
    I = eye(size(U));
elseif flag == 1
    I = eye(m);
end
Q = house_apply(U,I);
if flag == 1
    R = R(:,(1:n1));
else
    R = R((1:n1),(1:n1));
end
end


function u = house_gen(x)
% u = house_gen(x)
% Generate Householder reflection.
% u = house_gen(x) returns u with norm(u) = sqrt(2), and
% H(u,x) = x - u*(u'*x) = -+ norm(x)*e_1.

% Modify the sign function so that sign(0) = 1.
sig = @(u) sign(u) + (u==0);
nx = max(abs(x));
x1 = chop(x/nx);
nu = chop(nx*chop(sqrt(ip_chop(x1,x1))));
if nu ~= 0
    u = chop(x/nu);
    u(1) = chop(u(1) + sig(u(1)));
    u = chop(u/sqrt(abs(u(1))));
else
    u = x;
    u(1) = chop(sqrt(2));
end
end

function Z = house_apply(U,X)
% Apply Householder reflections.
% Z = house_apply(U,X), with U from house_qr
% computes Q*X without actually computing Q.
H = @(u,x) chop(x - chop(u*(hgemm(u',x))));
Z = X;
[~,n] = size(U);
for j = n:-1:1
    Z = H(U(:,j),Z);
end
end
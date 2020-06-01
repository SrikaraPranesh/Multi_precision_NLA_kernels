function [Q, R] = gs_m_chop(A,format)
%GS_M_CHOP    Modified Gram-Schmidt QR factorization using chop.
%        [Q, R] = GS_M_chop(A) uses the modified Gram-Schmidt method to compute the
%        factorization A = Q*R for m-by-n A of full rank,
%        where Q is m-by-n with orthonormal columns and R is n-by-n.

%        Reference:
%        N. J. Higham, Accuracy and Stability of Numerical Algorithms,
%        Second edition, Society for Industrial and Applied Mathematics,
%        Philadelphia, PA, 2002; sec 19.8.
fp.format = format; chop([],fp);
[m, n] = size(A);
Q = zeros(m,n);
R = zeros(n);

for k=1:n
    R(k,k) = chop(sqrt(ip_chop(A(:,k),A(:,k))));
    Q(:,k) = chop(A(:,k)/R(k,k));
    if k~=n
        R(k,k+1:n) = chop(Q(:,k)'*A(:,k+1:n));
        A(:,k+1:n) = chop(A(:,k+1:n) - chop(Q(:,k)*R(k,k+1:n)));
    end
end

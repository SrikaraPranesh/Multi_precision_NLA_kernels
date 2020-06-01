function [Q,R] = qr_multi_prec(A,nb)
%QR_MULTI_PREC Performs multi-precision QR factorization
%   i. Panel Factorization in single.
%   ii. Update in low precision using compensation
%   nb -- panel size
%
%   Note -- Chop must be MATLAB path

% convert to single precision
A = single(A);
[m,n] = size(A);
if nb>=n 
    error('Panel size should be less than number of columns');
elseif m<n
    error('Matrix should have more rows than columns');
end

cl = 1; cu = nb;
rl = 1; Q = single(eye(m,n)); R = single(zeros(n,n));
while cu<n 
    % panel factorization
    [Q(:,(cl:cu)),R((cl:cu),(cl:cu))] = qr(A(:,(cl:cu)),0);
    
    % Trailing submatrix update
    R((cl:cu),(cu+1:n)) = c_gemm(Q(:,(cl:cu))',A(:,(cu+1:n)));
    A(:,(cu+1:n)) = A(:,(cu+1:n))-c_gemm(Q(:,(cl:cu)),R((cl:cu),(cu+1:n)));

    rl = rl+nb; cl = cl+nb; cu = cu+nb;
    if cu >= n
        [Q(:,(cl:n)),R((cl:n),(cl:n))] = qr(A(:,(cl:n)),0);
    end
end




end


function [L,U,Pa] = cgemm_lu(A,nb)
%CGEMM_LU Summary of this function goes here
% LU factorization using partitioned LU algorithm, where the GEMM update is
% performed using compensated GEMM. The output L, U factors are in single
% precision.
%   A -- Input matrix
%   nb -- number of columns in each block. Default value is 10. 
% 
%
% Note -- Works only if the matrix size is greater than 20.


% make sure the input matrix is in single precision.
A = single(A);
n = length(A); % size of the matrix


if nargin < 2
    nb = 10; % number of columns in each block
end

% if mod(n,nb) ~= 0 
%    error('n is not divisible by nb'); 
% end

nbl = ceil(n/nb); %number of blocks

L = single(zeros(n)); U = single(zeros(n));
Pa = eye(n);

for i = 1:nbl
    i1 = ((i-1)*nb)+1;
    i2 = i*nb;
    
    if i2 > n
       i2 = n; 
    end
    
    [L((i1:n),(i1:i2)),U((i1:i2),(i1:i2)),P1] = lu(A((i1:n),(i1:i2)));    
    A((i1:n),:) = P1*A((i1:n),:);
    
    % apply the new permutation to earlier computations
    if i1 ~= 1
       L((i1:n),(1:i1-1)) = P1*L((i1:n),(1:i1-1));
    end
    
    % update the permutation matrix
    Pa((i1:n),:) = P1*Pa((i1:n),:);
    
    if i ~= nbl
        U((i1:i2),(i2+1:n)) = L((i1:i2),(i1:i2))\A((i1:i2),(i2+1:n));
        T1 = L((i2+1:n),(i1:i2)); T2 = U((i1:i2),(i2+1:n));
        C = matmulmulticomp(T1,T2);
        A((i2+1:n),(i2+1:n)) = A((i2+1:n),(i2+1:n))-C;
    end
end


end


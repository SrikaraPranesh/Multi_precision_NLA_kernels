function [R] = tc_block_chol(A,nb)
%TC_BLOCK_CHOL computes the Cholesky factorization using simulated
%              block FMA for Schur update step.
%   A -- Input square symmetric positive definite matrix
%   nb -- number of columns per block
%   R -- upper triangular Cholesky factor
%   NVIDIA tensor cores are simulated here, hence size of block FMA is 4
%   Algorithms used here is from http://www.netlib.org/utk/papers/factor/node9.html

[m,n] = size(A);
A = single(A);

if nargin<2 
    nb = 10; 
end

if m~=n, error('matrix is not square'), end
R = single(zeros(n));
if n <= nb 
    R = chol(A); 
    return; 
end
if norm(A-A')~=0, error('matrix is not symmetirc'), end
if min(eig(A))<0, error('matrix is not positive definite'), end



for i = 1:ceil(n/nb)
    l1 = ((i-1)*nb)+1;
    l2 = i*nb;
    
    if l2 >= n
       l2 = n; 
    end
    
    R((l1:l2),(l1:l2)) = chol(A((l1:l2),(l1:l2)));
    if l2~= n
        R((l1:l2),((l2+1):n)) = (R((l1:l2),(l1:l2)))'\A((l1:l2),((l2+1):n));
        
        %% Call to emulated tensor cores
        % pad the matrices with zeros to makes there dimensions multiple of
        % four.
        [T1,T2,~,T3] = bfmapadding(R((l1:l2),((l2+1):n))',...
                    R((l1:l2),((l2+1):n)),4,A(((l2+1):n),((l2+1):n)));
        
        % call to block FMA routine
        C = matmuladd(-T1,T2,T3);
        
        % remove extra padding
        [n_rows,~] = size(A(((l2+1):n),((l2+1):n)));
        A(((l2+1):n),((l2+1):n)) = C((1:n_rows),(1:n_rows));
       
        %% Usual Schur update
%         A(((l2+1):n),((l2+1):n)) = A(((l2+1):n),((l2+1):n)) - ...
%             (R((l1:l2),((l2+1):n))'*R((l1:l2),((l2+1):n)));
    end
end

end


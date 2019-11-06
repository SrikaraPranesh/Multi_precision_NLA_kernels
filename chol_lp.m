function [R,flag] = chol_lp(Ahp,format)
%CHOL_LP This function computes low precision Cholesky factorization
%   This functions computes the Cholesky factorization of a matrix in
%   two low precision, (i) fp16, and (ii) bfloat16. This function
%   uses the chop.m function from https://github.com/higham/chop.
%   Ahp -- Input matrix, should be in single or double precision
%   format -- 'h' : fp16
%             'b' : bfloat16
%             default value of format is 'h'

if nargin == 1, format = 'h'; end
u = float_params(format);

if (norm(Ahp-Ahp')~=0)
    error('Input matrix is not symmetric');
end

if (format ~= 'h' && format ~= 'b')
   error('unsupported low precision format') 
end
flag = 0;
n = length(Ahp);
fp.format = format; fp.round = 1;
A = chop(Ahp,fp); % convert to low precision

for k = 1:n
    big = A(k,k);  m = k;
    
    if big <= 0
        flag = 1;
        break;
    end
    
    
    if big == 0
        if norm(A(k+1:n,k)) ~= 0
            I = k; break
        else
            continue
        end
    end
    
    A(k,k) = chop(sqrt( A(k,k) ));
    if k == n, break, end
    A(k, k+1:n) = chop(A(k, k+1:n) / A(k,k));
    
    %   For simplicity update the whole of the remaining submatrix (rather
    %   than just the upper triangle).
    
    j = k+1:n;
    A(j,j) = chop(A(j,j) - chop(A(k,j)'*A(k,j)));
    
end

if flag == 0
    R = triu(A);
else
    R = zeros(size(A));
end

end

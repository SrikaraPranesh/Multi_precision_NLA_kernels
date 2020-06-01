function Ch = c_gemm(A,B)
%C_GEMM performs matrix-matrix multiplication with higher
%accuracy but with low-precision data representation.
%   Input matrices will be converted to fp16 and output is given
%   in fp32. 
%
%   Note -- chop function must be in Matlab path
[m,k] = size(A);
[k,n] = size(B);

As = single(A);
Bs = single(B);

Ah = cell(3,1);
Ah{1,1} = chop(As);
Ah{2,1} = chop(As-Ah{1,1});
Ah{3,1} = chop(As-Ah{1,1}-Ah{2,1});

Bh = cell(3,1);
Bh{1,1} = chop(Bs);
Bh{2,1} = chop(Bs-Bh{1,1});
Bh{3,1} = chop(Bs-Bh{1,1}-Bh{2,1});

Ch = zeros(m,n); % Half precision Tensor cores with single precision output
for i=1:3
    for j=1:3
        Ch = Ch + Ah{i,1}*Bh{j,1};
    end
end

end


function [B,D1,D2] = scale_diag_2side(A)
%SCALE_DIAG_2SIDE  Two-sided diagonal scaling of matrix.
%   [B,D1,D2] = scale_diag_2side(A) computes B = D1*A_D2 where the diagonal 
%   matrices D1 = diag(d1) and D2 = diag(d2) are chosen so that the
%   largest elements in absolute value in every row and every column of B is 1.
%   If A has a zero row or column it will stay zero.

d1 = 1./max(abs(A),[],2);
B = diag(d1)*A;  
d2 = 1./max(abs(B),[],1);
B = B*diag(d2);

D1=diag(d1); D2=diag(d2);
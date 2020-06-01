function [x] = ip_chop_compensate(a,b)
%IP_CHOP_compensate computes inner product using chop function
%and compensation. Therefore it uses low-precision data format
%but results are as accurate as single precision
%   Note -- Requires chop.m to be in the MATLAB path


as = single(a);
bs = single(b);

ah = cell(3,1);
ah{1,1} = chop(as); ah{2,1} = chop(as-ah{1,1}); ah{3,1} = chop(as-ah{1,1}-ah{2,1});
bh = cell(3,1);
bh{1,1} = chop(bs); bh{2,1} = chop(bs-bh{1,1}); bh{3,1} = chop(bs-bh{1,1}-bh{2,1});

x = 0;
for i=1:3
    for j=1:3
        x = x+ah{i,1}'*bh{j,1};
%         x = x+ip_chop(ah{i,1},bh{j,1});
    end
end

end


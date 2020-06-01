function varargout = bfmapadding(Ain,Bin,bfma,Cin)
% BFMAPADDING Pad matrices for use with MATMULADD.
%   [AOUT,BOUT,COUT] = BFMAPADDING(AIN,BIN,CIN,BFMA) adds trailing rows and
%   columns of zeros to AIN, BIN, and CIN so that the dimensions of AOUT,
%   BOUT, and COUT are integer multiples of BFMA(1) and BFMA(2), BFMA(2) and
%   BFMA(3), and BFMA(1) and BFMA(3), respectively.

  if nargin == 3
    [conformable,sizes] = bfmaconformable(Ain,Bin);
  else
    [conformable,sizes] = bfmaconformable(Ain,Bin,Cin);
  end
  if conformable
    new_sizes = bfma .* ceil(sizes ./bfma);
    if any(sizes(1:2) ~= new_sizes(1:2))
      varargout{1} = ...
          [Ain zeros(sizes(1), new_sizes(2)-sizes(2), 'like', Ain);
           zeros(new_sizes(1)-sizes(1), new_sizes(2), 'like', Ain)];
    else
      varargout{1} = Ain;
    end
    if any(sizes(2:3) ~= new_sizes(2:3))
      varargout{2} = ...
          [Bin zeros(sizes(2), new_sizes(3)-sizes(3), 'like', Bin);
           zeros(new_sizes(2)-sizes(2), new_sizes(3), 'like', Bin)];
    else
      varargout{2} = Bin;
    end
    varargout{3} = new_sizes;
    if nargin == 4
      if any(sizes([1,3]) ~= new_sizes([1,3]))
        varargout{4} = ...
            [Cin zeros(sizes(1), new_sizes(3)-sizes(3), 'like', Cin);
             zeros(new_sizes(1)-sizes(1), new_sizes(3), 'like', Cin)];
      else
        varargout{4} = Cin;
      end
    end
  else
    error("bfmapadding:wrongMatrixDimensions",...
          "The matrices are not conformable");
  end
end
function T = bintruncate(A, fmin, fmax, t)
% BINTRUNCATE Truncate binary representation of floating point numbers.
%  T = BINTRUNCATE(A,FMIN,FMAX,P) computes a matrix T of the same size as A
%  such that
%    - T(i,j) = 0,    if abs(A(i,j)) < FMIN
%    - T(i,j) = sign(A(i,j)) * Inf,     if abs(A(i,j)) > FMAX
%    - T(i,j) = sign(A) * 2^(-p) * floor(2^p * abs(A(i,j))),    otherwise.

  if nargin < 4
    [~,options] = chop;
    [u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(options.format);
    if nargin < 2
      fmin = xmin;
    end
    if nargin < 3
      fmax = xmax;
    end
    if nargin < 4
      t = p - 1;
    end
  end
  T = A;
  T(abs(T)<fmin) = 0;
  indices = abs(T)>fmax;
  T(indices) = sign(T(indices)) * Inf;
  indices = abs(T) <= fmax & abs(T) >= fmin;
  T(indices) = sign(T(indices)) .* chop(2^-t* floor(2^t * abs(T(indices))));
end
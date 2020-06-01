function [Aexp, q] = scaledexpansion(A)
% SCALEDEXPANSION compute a scaled expansion of A in terms of lower precision.
  R = A;
  Rold = R;
  q = 0;
  done = false;
  [~,options] = chop;
  [u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(options.format);
  while ~done
    q = q + 1;
    Aexp(:,:,q) = bintruncate(R,xmin,xmax,p-1);
    R = 2^p*(R - Aexp(:,:,q));
    done = all(R(:) == 0) || all(R(:) == Rold(:));
    Rold = R;
  end
end
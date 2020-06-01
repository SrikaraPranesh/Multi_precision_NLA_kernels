function [xMR, iflag, total_iters, resMR] = lsqminres_dq(A,Q,R,f,a,tol,n_max,x0,ptype)
% LSQMINRES_DQ     preconditioned MINRES algorithm in double and quad precision
% [xMR, iflag, total_iters, resMR] = lsqminres_dq(A,Q,R,f,a,tol,n_max,x0)
%          Used for solving the correction equation in the iterative
%                 refinement of least squares problem using augmented matrix.
%   input
%          A            Input n X n symmetric matrix
%          Q,R          Q and R factors of A to be used in preconditioners
%          f            right-hand side vector
%          a            scaling factor for augmented matri
%          n_max        max number of iterations
%          tol          relative preconditioned residual reduction factor            
%          x0           initial iterate
%          ptype        1 for low precision inverse
%                       2 for block diagonal 
%   output
%          x            computed (approximate) solution
%          iflag        convergence flag
%                       0 for convergence to tol within n_max iterations
%                       1 if iterated n_max times but did not converge to tol
%          total_iters  total iteration count
%          resMR        vector of residual norms of iterates
%   PIFISS function: DJS; 31 January 2007

%Modification of pminres.m written by Bernd Fischer.
%Copyright 1996 B. Fischer.  
%Reproduced and distributed with permission of the copyright holder.
%Modifications to handle PIFISS data structures identified in line.

% This function is modified by Srikara Pranesh. 31 May 2020.
% note -- required Advanpix toolbox 

% convert input data to double precision
A = double(A); Q = double(Q); R = double(R);
f = double(f); a = double(a);

precond=1;
% Names of mvp and preconditioning routines
   
%%Initialize
   n=1; N=length(x0); 
      v_hat_old=zeros(N,1); %v_hat=f-feval(matvec,A,x0);
      v_hat=f-A_fun(A,a,x0);
	  norm_r0=norm(v_hat);
      %y=feval(precondsolver,L,nedges,nel,v_hat, grid_data, smoother_data);
      y=M_fun(Q,R,a,v_hat,ptype);
      beta=sqrt(v_hat'*y);
      beta_old=1;
      % take preconditoned residual error
      %norm_r0=beta;   
   
   c=1; c_old=1; s_old=0; s=0; 
   w=zeros(N,1); w_old=w; eta=beta;
   xMR=x0;  norm_rMR=norm_r0; norm_rGAL=norm_r0; 
   resMR=zeros(n_max,1); resGAL=zeros(n_max,1); 
   resMR(n)=log10(norm_rMR/norm_r0);
   resGAL(n)=log10(norm_rGAL/norm_r0);
   
   while (n < n_max+1) & (norm_rMR/norm_r0 > tol)  
      n=n+1;
   %%Lanczos
	     v=y/beta; 
         y=v_hat;        
		 %Av=feval(matvec,A,v); 
         Av=A_fun(A,a,v);
         alpha=v'*Av;
		 beta;
         v_hat=Av-(alpha/beta)*v_hat-(beta/beta_old)*v_hat_old;
         v_hat_old=y; 
       
         %y=feval(precondsolver,L,nedges,nel,v_hat,grid_data,smoother_data)
         y=M_fun(Q,R,a,v_hat,ptype);
         
         beta_old=beta; beta=sqrt(v_hat'*y);
         

   %%QR factorization
      c_oold=c_old; c_old=c; s_oold=s_old; s_old=s;
      

      r1_hat=c_old*alpha-c_oold*s_old*beta_old;
      r1    =sqrt(r1_hat^2+beta^2);
      r2    =s_old*alpha+c_oold*c_old*beta_old;
      r3    =s_oold*beta_old;

   %%Givens rotation
      c=r1_hat/r1;
      s=beta/r1; 
   
   %%Update
      w_oold=w_old; w_old=w; 
      w=(v-r3*w_oold-r2*w_old)/r1 ;
      xMR=xMR+c*eta*w; eta=-s*eta;
	  
   %%CEP - Experiment with different stopping criteria if required
	 %precres1=(f - feval(matvec, A,xMR));
     %precres2=feval(precondsolve,L,R,nedges,nel,precres1);
     %precres3=precres1'*precres2;
     %size(precres3);
     %norm_prec_res=sqrt(precres3);
     %norm_rMR=norm_prec_res;
     %norm_rMR=norm(newres2)
     
     norm_rMR=norm_rMR*abs(s) ; %Updated norm w.r.t. preconditioned system
	 
     %norm_rMR=norm(f-feval( matvec,A,xMR));
     % norm_rMR=beta;
          resMR(n)= norm_rMR/norm_r0; %% log10(norm_rMR/norm_r0);
	
	  if abs(c) > eps
         xGAL=xMR-s*eta*w/c; 
		 
	    %%Norm
	     norm_rGAL=norm_rMR/abs(c); %Updated norm w.r.t. preconditioned system
	     %norm_rGAL=norm(f-feval( matvec,A,xGAL));
         resGAL(n)=log10(norm_rGAL/norm_r0);
	    else
 
        disp([int2str(n),' no Galerkin iterate exists']);
      end; %Galerkin
      
    end %while	
	resMR =resMR(1:n);
    resGAL=resGAL(1:n);

    iflag=0; total_iters=n;
%%% set flag if iteration count exceeded 
    if n==n_max+1, iflag = 1; end
 
return;

function x = A_fun(A,a,d)
%A_fun Matrix-Vector product with the augmented system
%       x = A_fun(A,a,d)
%       A -- real mxn matrix where m>n, and full rank.
%       d -- real (m+n)x1 input vector.
%       a -- scaling factor used in augmented system.
mp.Digits(34);
[m,n] = size(A);
A = mp(A);
a = mp(a); d = mp(d);
r = d((1:m),1); s = d((m+1:(m+n)),1);

t1 = a*r; t2 = A*s;
t3 = A'*r;
x = double([t1+t2;t3]);

function x = M_fun(Q,R,a,b,ptype)
%M_fun Application of augmented matrix inverse constructed using QR factors
%      x = M_fun(Q,R,a,b,ptype)
%       Q -- real mxn orthogonal matrix.
%       R -- real nxn upper triangulat matrix.
%       d -- real (m+n)x1 input vector.
%       a -- scaling factor used in augmented system.
%       ptype -- 2 selects the block diagonal preconditioner.
mp.Digits(34);
[m,n] = size(Q);
Q = mp(Q); R = mp(R);
a = mp(a); b = mp(b);
r = b((1:m),1); s = b((m+1:(m+n)),1);

if ptype == 1
  t1 = (r-(Q*(Q'*r)))/a;
  t2 = Q*(R'\s);
  t3 = R\(Q'*r);
  t4 = -a*(R\(R'\(s)));
  x = double([t1+t2;t3+t4]);
elseif ptype == 2
  t1 = r/a;
  t4 = a*(R\(R'\(s)));
  x = double([t1;t4]);
end 






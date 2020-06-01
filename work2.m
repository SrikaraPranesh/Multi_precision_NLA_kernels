clear all
close all

A = gallery('randsvd',23,-1e5);
R = tc_block_chol(A,5);

norm((A-(R'*R)),inf)/norm(A,inf)
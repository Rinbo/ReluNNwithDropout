## Copyright (C) 2017 robin


## Author: robin <robin@XPS15>
## Created: 2017-05-28

function [R] = reluDerivative (z)

R = (z>=0);
R = double(R);
R(find(R==0)) = 0.01;

endfunction

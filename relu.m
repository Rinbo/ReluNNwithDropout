## Copyright (C) 2017 robin

## Author: robin <robin@XPS15>
## Created: 2017-05-28

function [R] = relu (z)

R = max(z,0);
k = find(R==0);
R(k) = z(k)*0.01;
%relu = (z >= 0);
%retval = z.*relu;

endfunction

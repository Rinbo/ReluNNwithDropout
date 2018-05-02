## Copyright (C) 2017 robin

## Author: robin <robin@XPS15>
## Created: 2017-05-31

function [SM] = softmax (Z)
% Modified softmax to avoid numerical underflow:
Z = Z-max(Z);
Z = exp(Z);
SM = Z./sum(Z);
endfunction

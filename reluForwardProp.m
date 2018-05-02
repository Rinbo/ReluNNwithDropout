## Copyright (C) 2017 robin

## Author: robin <robin@XPS15>
## Created: 2017-06-06
% This function assumes you have added bias to input matrix
function [H, A1, A2] = reluForwardProp (X, Theta1, Theta2, Theta3, M1, M2)

if nargin == 4,
	M1 = 1;
	M2 = 1;
end

A1 = relu(Theta1*X).*M1; 
A1 = [ones(1,size(A1,2));A1]; 
A2 = relu(Theta2*A1).*M2;
A2 = [ones(1,size(A2,2));A2];
H = softmax(Theta3*A2);



endfunction

## Copyright (C) 2017 robin

## Author: robin <robin@XPS15>
## Created: 2017-06-06
% Assumes X contains biases and that Y is a binary matrix

function [D1, D2, D3] = reluBackProp (X, YB, Theta1, Theta2, Theta3, M1, M2)

if nargin == 5,
	M1 = 1;
	M2 = 1;
end

m = size(X,2);

% Forward Prop
[H, A1, A2] = reluForwardProp(X, Theta1, Theta2, Theta3, M1, M2);
		
delta3 = H-YB; 
D3 = (1/m)*(delta3*A2'); 
delta2 = Theta3'*delta3.*reluDerivative(A2); 
delta2 = delta2(2:end, :).*M1;
D2 = (1/m)*(delta2*A1');
delta1 = Theta2'*delta2.*reluDerivative(A1);
delta1 = delta1(2:end, :).*M2;
D1 = (1/m)*(delta1*X');

endfunction

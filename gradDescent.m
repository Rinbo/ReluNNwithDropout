## Copyright (C) 2017 robin

## Author: robin <robin@XPS15>
## Created: 2017-06-06

function [Theta1, Theta2, Theta3] = gradDescent (Theta1, Theta2, Theta3, D1, D2, D3, alpha)

Theta1 = Theta1 - alpha*D1;
Theta2 = Theta2 - alpha*D2;
Theta3 = Theta3 - alpha*D3;



endfunction

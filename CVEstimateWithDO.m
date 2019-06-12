## Copyright (C) 2017 robin
## Author: robin <robin@XPS15>
## Created: 2017-06-26

function H = CVEstimateWithDO (X, Theta1, Theta2, Theta3)

  A1 = relu(Theta1*X); 
  A1 = [ones(1,size(A1,2));A1]; 
  A2 = relu(Theta2*A1);
  A2 = [ones(1,size(A2,2));A2];
  H = softmax(Theta3*A2); 

endfunction

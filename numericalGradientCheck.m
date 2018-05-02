% Script to check if the gradient computation of the given backprop function works

epsilon = 0.0001;
m = 25; % Number of trainings examples
XCHECK = X(:, 1:m); %Only choose a subset of the training examples
YCHECK = Y(1:m);

% Initialize network hyperparameters

num_hidden_units = 5; % Number of hidden Units : 25
num_output_units = size(unique(YCHECK), 1);
num_input_units = size(XCHECK(:,1),1);

% Initialize Theta-matrices:

eps1 = sqrt(2/num_input_units);
eps2 = sqrt(2/num_hidden_units);

Theta1 = randn(num_hidden_units, num_input_units+1)*eps1; 
Theta2 = randn(num_hidden_units, num_hidden_units+1)*eps2; 
Theta3 = randn(num_output_units, num_hidden_units + 1)*eps2;

% Add Bias to X matrix

XCHECK = [ones(1,m);XCHECK];

% Convert Y to a binary matrix
YBCHECK = zeros(num_output_units, m);
for i = 1:m,
	if YCHECK(i) == 0, YCHECK(i) = 10; end; % 10 = 0
	YBCHECK(YCHECK(i),i) = 1;
end;

% Frist calculate the gradient using our backprop script:

[D1, D2, D3] = reluBackProp (XCHECK, YBCHECK, Theta1, Theta2, Theta3);

% Now lets calculate the gradient using numerical gradient checking

[numGradD1, numGradD2, numGradD3] = numGrad (Theta1, Theta2, Theta3, XCHECK, YBCHECK, epsilon);

C1 = D1-numGradD1;
C2 = D2-numGradD2;
C3 = D3-numGradD3;
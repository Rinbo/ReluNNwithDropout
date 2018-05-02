## Copyright (C) 2017 robin
## Author: robin <robin@XPS15>
## Created: 2017-05-30

function [Theta1, Theta2, Theta3, accCV] = relu2NN (X, Y, XCV, YCV, ...
												epoch, alpha)

% Network Architecture:
% Number of hidden Layers: 2
num_hidden_units = 4000; % Number of hidden Units
num_output_units = size(unique(Y), 1);
num_input_units = size(X(:,1),1);
m = size(X,2); % Number of trainings examples
mm = size(XCV,2); % Number of CV examples
batchSize = 100;
%eps = 0.3; % Small number to avoid vanishing graident

% Initialize Theta and bias to input matrix
eps1 = sqrt(2/num_input_units);
eps2 = sqrt(2/num_hidden_units);
%eps2 = 4*sqrt(6/(num_hidden_units*2));
Theta1 = randn(num_hidden_units, num_input_units+1)*eps1; 
Theta2 = randn(num_hidden_units, num_hidden_units+1)*eps2; 
Theta3 = randn(num_output_units, num_hidden_units + 1)*eps2;

X = [ones(1,m);X];
XCV = [ones(1,mm);XCV];

% Convert Y to a binary matrix
YB = zeros(num_output_units, m);
for i = 1:m,
	if Y(i) == 0, Y(i) = 10; end; % 10 = 0
	YB(Y(i),i) = 1;
end;

YCVB = zeros(num_output_units, mm);
for i = 1:mm,
	if YCV(i) == 0, YCV(i) = 10; end; % 10 = 0
	YCVB(YCV(i),i) = 1;
end;

% Minibtach formulation
miniBatchSize = 50;
iterationsPerEpoch = m/miniBatchSize;

accCV = zeros(epoch,1);

% The Algorithm:
for i = 1:epoch,
	for j = 1:iterationsPerEpoch,
		%Divide into batch
		XMB = X(:,(miniBatchSize*(j-1)+1):(j*miniBatchSize));
		YMB = YB(:,(miniBatchSize*(j-1)+1):(j*miniBatchSize));
		
		% Backward prop
		% Also pass in mask
		M1 = binornd(1, 0.5, num_hidden_units, miniBatchSize);
		M2 = binornd(1, 0.5, num_hidden_units, miniBatchSize);
		[D1, D2, D3] = reluBackProp(XMB, YMB, Theta1, Theta2, Theta3, M1, M2);
		% Gradient Descent:
		[Theta1, Theta2, Theta3] = gradDescent (Theta1, Theta2, Theta3, ...
										D1, D2, D3, alpha);
	end
	%H = reluForwardProp(X, Theta1, Theta2, Theta3);
	
	% For future ref: standalone function for cost and acc
	%J(i) = -(1/m)*(sum(sum(YB.*log(H))));
	%acc(i) = performance (H, Y);
	%JCV(i) = -(1/mm)*(sum(sum(YCVB.*log(HCV))));
	
	% Do early stopping
	HCV = CVEstimateWithDO (XCV, Theta1*0.5, Theta2*0.5, Theta3*0.5);
	accCV(i) = performance (HCV, YCV);
	
	printf("Epoch number: %d\n", i);
	fflush(stdout);
	%if i > 40,
	%	if sum(JCV(i-10:end))/11 < JCV(i),
	%		break;
	%	end
	%end
end


endfunction

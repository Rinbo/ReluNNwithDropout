# Neural Net with Dropout

- Created: 2017-05-30
- Published: 2018-03-30
- Refactored: 2019-06-23

This neural network is of a standard architecture with 2 hidden layers. The number of nodes in the hidden layers is customizable. The same goes for the minibatch size, epochs and of course alpha (learning rate). Activation function used is Relu (leaky). Optimzation method is mini-batch gradient descent.

Regularization method used is dropout (50%). Using this network with the below hyperparameters and one week of continous calculations on my desktop computer yeilded a result of 98.02% accuracy on the cross validation set.

60K training images/10K CV

Hidden layers: 2
Hidden units per layer: 4096
Alpha: 0.003 Batch size: 100 Reg method: Dropout (p = 0.5) Iterations: 1500 epochs

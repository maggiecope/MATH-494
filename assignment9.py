# m494s24 hw9
#
# Maggie Cope! 
#
# Toy neural network closely based on the code in the book Neural
# Network Projects with Python, by James Loy.
#
# You will input four bits (thus four input nodes) and output one
# value. You may use an arbitrary number of hidden nodes. The task
# is to teach the network to count the number of ones (e.g., 0111
# should yield 3). With complete input,that is 16 patterns, this is
# an easy task, so you will try omitting some patterns and check
# whether the network generalized.
#
# m494s24 April 2024

import numpy as np
import matplotlib.pylab as plt

eta = 0.1
seedy = 42
NUM_HIDDEN = 4

def sigmoid(x):
    
    # Sigmoid activation function.
    
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):

    # Derivative of the sigmoid function./solace-london-bysha-maxi-dress-in-bubblegum/dp/SOLA-WD186/

    return x * (1.0 - x)

class ToyNN:
    def __init__(self, x, y):
        
        #Initialize the neural network.
        
        self.input = x
        self.wts1 = np.random.rand(self.input.shape[1], NUM_HIDDEN)
        self.wts2 = np.random.rand(NUM_HIDDEN, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):

        # Perform the feedforward step.
        
        self.layer1 = sigmoid(np.dot(self.input, self.wts1))
        self.output = sigmoid(np.dot(self.layer1, self.wts2))

    def backprop(self):

        # Perform the backpropagation step.
        
        diff_wts2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        diff_wts1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.wts2.T) * sigmoid_derivative(self.layer1)))
        self.wts1 += eta*diff_wts1
        self.wts2 += eta*diff_wts2

    def check(self, input1):
        
        # Check the output of the neural network for a given input.

        self.layer1 = sigmoid(np.dot(input1, self.wts1))
        self.output = sigmoid(np.dot(self.layer1, self.wts2))

if __name__ == "__main__":
    # Enter the set of input patterns here:
    X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
                  [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                  [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                  [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    
    # Enter the set of corresponding outputs
    y = np.array([[0], [1], [1], [2], [1], [2], [2], [3], [1], [2], [2], [3], [2], [3], [3], [4]])
    y = y / 4  # Scale the output values to the range (0, 1)

    np.random.seed(seedy)

    print("Parameters:")
    print("Learning rate (eta):", eta)
    print("Number of hidden nodes:", NUM_HIDDEN)
    print("PRNG seed:", seedy)

    # Train with complete input
    print("\nTraining with complete input:")
    nn = ToyNN(X, y)
    for j in range(5000):
        nn.feedforward()
        nn.backprop()

    # Checking the output of patterns:
    errors = []
    for i in range(len(X)):
        nn.check(X[i])
        result = 4*nn.output
        target = 4*y[i]
        error = abs(result - target) / target
        errors.append(error)
        print(f"Input: {X[i]}, Target: {target[0]:.2f}, Output: {result[0]:.2f}, Relative Error: {error[0]:.2%}")
    avg_error = np.mean(errors)
    print(f"\nAverage Relative Error: {avg_error:.2%}")

    # Train with one pattern omitted
    omitted_pattern = [1, 1, 1, 0]
    print(f"\nTraining with pattern {omitted_pattern} omitted:")
    X_omitted = np.delete(X, 14, axis=0)
    y_omitted = np.delete(y, 14, axis=0)
    nn_omitted = ToyNN(X_omitted, y_omitted)
    for j in range(5000):
        nn_omitted.feedforward()
        nn_omitted.backprop()
    nn_omitted.check(omitted_pattern)
    result_omitted = 4*nn_omitted.output
    print(f"Omitted Pattern: {omitted_pattern}, Expected Output: 3.00, Actual Output: {result_omitted[0]:.2f}")

    # Train with two patterns omitted
    omitted_patterns = [[1, 1, 1, 0], [0, 1, 1, 0]]
    print(f"\nTraining with patterns {omitted_patterns[0]} and {omitted_patterns[1]} omitted:")
    X_omitted_2 = np.delete(X, [6, 14], axis=0)
    y_omitted_2 = np.delete(y, [6, 14], axis=0)
    nn_omitted_2 = ToyNN(X_omitted_2, y_omitted_2)
    for j in range(5000):
        nn_omitted_2.feedforward()
        nn_omitted_2.backprop()
    for pattern in omitted_patterns:
        nn_omitted_2.check(pattern)
        result_omitted_2 = 4*nn_omitted_2.output
        print(f"Omitted Pattern: {pattern}, Expected Output: {4*y[np.where((X==pattern).all(axis=1))[0][0]][0]:.2f}, Actual Output: {result_omitted_2[0]:.2f}")
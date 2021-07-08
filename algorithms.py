import numpy as np
import random
import time
from math import exp
from random import seed
from random import random

class Algorithms():
    class Perceptron():
        def __init__(self):
            """
            Perceptron constructor
            Seed the random number generator
            Set weights to a 3x1 matrix (input shape)
            With values from -1 to 1 and mean 0
            """
            np.random.seed(int(time.time()))
            self.weights = 2 * np.random.random((3, 1)) - 1

        def sigmoid(self, x):
            """
            Returns the sigmoid of x
                Parameters:
                    x (ndarray): input numpy array of floats
                Returns:
                    sigmoid(x)(ndarray): normalized weighted sum of the inputs
            """
            return 1 / (1 + np.exp(-x))

        def sigmoid_dx(self, x):
            """
            Returns the derivative of the sigmoid, used for weight adjustments
                Parameters:
                    x (ndarray): numpy array of floats
                Returns:
                    sigmoid_dx(x)(ndarray): numpy array of floats
            """
            return x * (1 - x)

        def train(self, X_train, y_train, training_iterations):
            """
            Trains the model and adjust its weights with each iteration
                Parameters:
                    X_train (ndarray): feature vector of ints
                    y_train (ndarray): label vector of ints
                    training_iterations (int): number of iterations
                Returns:
                    model (dict): trained model dictionary
            """
            for _ in range(training_iterations):
                output = self.results(X_train)
                error = y_train - output
                # Backpropagation: Error weighted derivatives
                adjustments = np.dot(X_train.T, error * self.sigmoid_dx(output))
                self.weights = self.weights + adjustments

            model = {'output':output,
                     'error':error,
                     'adjustments':adjustments,
                     'weights':self.weights,
                    }

            return model

        def results(self, inputs):
            """
            Pass inputs through the perceptron to get the output
                Parameters:
                    inputs (ndarray): feature vector of ints
                Returns:
                    output (ndarray): normalized weighted sum of the inputs via sigmoid
            """
            inputs = inputs.astype(float)
            output = self.sigmoid(np.dot(inputs, self.weights))
            return output

    class NeuralNetwork():
        def __init__(self):
            """
            NN constructor
            Seed the random number generator
            Set weights to a 3x1 matrix (input shape)
            With values from -1 to 1 and mean 0
            """
            np.random.seed(int(time.time()))
            self.weights = 2 * np.random.random((3, 1)) - 1

        def initialize_network(self, n_inputs, n_hidden, n_outputs):
            """
            Initializes the neural network with one hidden layer
                Parameters:
                    n_inputs (int): number of input layer nodes
                    n_hidden (int): number of hidden layer nodes
                    n_outputs (int): number of output layer nodes
                Returns:
                    network (list of lists): neural network
            """
            network = list()
            hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
            network.append(hidden_layer)
            output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
            network.append(output_layer)
            return network

        def activate(self, weights, inputs):
            """
            Calculate neuron activation for an input (Z = X * W + b)
                Parameters:
                    inputs (ndarray): feature vector
                Returns:
                    activation (ndarray): neuron activation
            """
            activation = weights[-1]
            for i in range(len(weights)-1):
                activation += weights[i] * inputs[i]
            return activation

        def transfer(self, activation):
            """
            Sigmoid activation function
                Parameters:
                    activation (ndarray): neuron activation
                Returns:
                    sigmoid activation
            """
            return 1.0 / (1.0 + exp(-activation))

        def forward_propagate(self, network, row):
            """
            Forward propagate input to a network output
                Parameters:
                    network (list of lists): neural network
                    row (list): input pattern
                Returns:
                    inputs (list): output of the forward pass
            """
            inputs = row
            for layer in network:
                new_inputs = []
                for neuron in layer:
                    activation = self.activate(neuron['weights'], inputs)
                    neuron['output'] = self.transfer(activation)
                    new_inputs.append(neuron['output'])
                inputs = new_inputs
            return inputs

        def transfer_derivative(self, output):
            """
            Sigmoid derivative
                Parameters:
                    output (float): output value of a neuron
                Returns:
                    sigmoid derivative
            """
            return output * (1.0 - output)

        def backward_propagate_error(self, network, expected):
            """
            Backpropagate error and store in neurons
                Parameters:
                    network (list of lists): neural network
                    expected (list): expected output values
                Returns:
                    void
            """
            for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                    for j in range(len(layer)):
                        error = 0.0
                        for neuron in network[i + 1]:
                            error += (neuron['weights'][j] * neuron['delta'])
                        errors.append(error)
                else:
                    for j in range(len(layer)):
                        neuron = layer[j]
                        errors.append(expected[j] - neuron['output'])
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

        def update_weights(self, network, row, l_rate):
            """
            Update network weights with error
                Parameters:
                    network (list of lists): neural network
                    row (list): each row of the training data
                    l_rate (float): learning rate
                Returns:
                    void
            """
            for i in range(len(network)):
                inputs = row[:-1]
                if i != 0:
                    inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                    for j in range(len(inputs)):
                        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                    neuron['weights'][-1] += l_rate * neuron['delta']

        def train_network(self, network, train, l_rate, n_epoch, n_outputs):
            """
            Train a network for a fixed number of epochs
                Parameters:
                    network (list of lists): neural network
                    train (list): training data
                    l_rate (float): learning rate
                    n_epoch (int): number of epochs
                    n_outputs (int): number of outputs
                Returns:
                    model (dict): trained model dictionary
            """
            for epoch in range(n_epoch):
                sum_error = 0
                for row in train:
                    outputs = self.forward_propagate(network, row)
                    expected = [0 for i in range(n_outputs)]
                    expected[row[-1]] = 1
                    sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                    self.backward_propagate_error(network, expected)
                    self.update_weights(network, row, l_rate)
                print('epoch=%d, lrate=%.3f, loss=%.3f' % (epoch, l_rate, sum_error))

            model = {'output':outputs,
                     'error':sum_error,
                     'weights':self.weights,
                    }

            return model

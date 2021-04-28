import numpy as np
import random
import time

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

        def train_test(self, X_test, y_test, training_iterations):
            """
            Trains the model and adjust its weights with each iteration
                Parameters:
                    X_test (ndarray): feature vector of ints
                    y_test (ndarray): label vector of ints
                    training_iterations (int): number of iterations
            """
            for _ in range(training_iterations):
                output = self.results(X_test)
                error = y_test - output
                # Backpropagation: Error weighted derivatives
                adjustments = np.dot(X_test.T, error * self.sigmoid_dx(output))
                self.weights = self.weights + adjustments

        def results(self, inputs):
            """
            Pass inputs through the perceptron to get the output
                Parameters:
                    inputs (ndarray): feature vector of ints
                Returns:
                    output (ndarray): normalized weighted sum of the inputs via sigmoid
            """
            inputs = inputs.astype(float)
            self.weights = self.weights.astype(float)
            output = self.sigmoid(np.dot(inputs, self.weights))
            return output

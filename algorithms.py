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

        def results(self, inputs):
            """
            Pass inputs through the perceptron to get the output
                Parameters:
                    inputs (ndarray): feature vector of ints
                Returns:
                    output (ndarray): normalized weighted sum of the inputs via sigmoid
            """
            inputs = inputs.astype(float)
            #self.weights = self.weights.astype(float)
            output = self.sigmoid(np.dot(inputs, self.weights))
            return output

    class NeuralNetwork(Perceptron):
        def __init__(self):
            pass

        def init_params(self):
            """
            Initialization of weights and biases
                Returns:
                    W1 (ndarray): first layer weights
                    b1 (ndarray): first layer biases
                    W2 (ndarray): second layer weights
                    b2 (ndarray): second layer biases
            """
            W1 = 2 * np.random.rand(3, 3) - 1
            b1 = 2 * np.random.rand(1, 3) - 1 # (3,1)
            W2 = 2 * np.random.rand(3, 1) - 1
            b2 = 2 * np.random.rand(1, 1) - 1
            return W1, b1, W2, b2

        def forward_prop(self, W1, b1, W2, b2, X):
            """
            Calculate the current loss
                Parameters:
                    W1 (ndarray): weights at first layer
                    b1 (ndarray): bias at first layer
                    W2 (ndarray): weights at second layer
                    b2 (ndarray): bias at second layer
                    X (ndarray): feature vector
                Returns:
                    Z1 (ndarray): unactivated layer 1
                    A1 (int): activated layer 1
                    Z2 (ndarray): unactivated layer 2
                    A2 (int): activated layer 2
            """
            Z1 = np.dot(X, W1) + b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = self.sigmoid(Z2)
            return Z1, A1, Z2, A2

        def error(self, target, output):
            """
            Returns the error
                Parameters:
                    target (float): target output
                    output (float): actual output A2
            """
            return 1/2 * (target - output)**2

        def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
            """
            Calculate current gradient
                Parameters:
                    Z1 (ndarray): unactivated layer 1
                    A1 (int): activated layer 1
                    Z2 (ndarray): unactivated layer 2
                    A2 (int): activated layer 2
                    W1 (ndarray): first layer weights
                    W2 (ndarray): second layer weights
                    X (list of ndarray): training data
                    Y (list of ndarray): testing data
                Returns:
                    dw1 (ndarray): gradients of W1
                    db1 (ndarray): gradients of b1
                    dw2 (ndarray): gradients of W2
                    db2 (ndarray): gradients of b2
            """
            Y = np.concatenate(Y, axis=0)
            Y = np.delete(Y, 0)
            Y = np.asarray(Y).reshape(-1, 1)
            m = Y.size
            dZ2 = self.error(A2, Y)
            print('************** INSIDE BACKPROP **************')
            print('A2: ', A2.shape)
            print(A2)
            print('Y: ', type(Y), Y.shape)
            print(Y)
            print('A2 - Y: ', (A2 - Y).shape)
            print(A2 - Y)
            print('dZ2: ', dZ2.shape)
            print(dZ2)
            print('A1: ', A1.shape)
            print(A1)
            print('A1.T: ', A1.T.shape)
            print(A1.T)
            dW2 = 1 / m * np.dot(dZ2.T, A1)
            db2 = 1 / m * np.sum(dZ2) 
            dZ1 = np.dot(W2.T, dZ2) * self.sigmoid_dx(Z1)
            dW1 = 1 / m * np.dot(dZ1, X.T)
            db1 = 1 / m * np.sum(dZ1)
            return dW1, db1, dW2, db2

        # server
        def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
            """
            Update Parameters
                Parameters:
                    W1 (ndarray): weights at first layer
                    b1 (ndarray): bias at first layer
                    W2 (ndarray): weights at second layer
                    b2 (ndarray): bias at second layer
                    dw1 (ndarray): gradients of W1
                    db1 (ndarray): gradients of b1
                    dw2 (ndarray): gradients of W2
                    db2 (ndarray): gradients of b2
                    alpha (float): learning rate
                Returns:
                    W1 (ndarray): updated W1
                    b1 (ndarray): updated b1
                    W2 (ndarray): updated W2
                    b2 (ndarray): updated b2
            """
            W1 = W1 - alpha * dW1
            b1 = b1 - alpha * db1
            W2 = W2 - alpha * dW2
            b2 = b2 - alpha * db2
            return W1, b1, W2, b2

        # server
        def get_predictions(self, A2):
            """
            Get predictions
                Parameters:
                    A2 (int): activation layer 2
                Returns:
                    Predictions
            """
            return np.argmax(A2, 0)

        # server
        def get_accuracy(self, predictions, Y):
            """
            Get accuracy
                Parameters:
                    predictions (ndarray): predictions
                    Y (ndaray): actual testing values
                Returns:
                    accuracy (float)
            """
            print(predictions, Y)
            return np.sum(predictions == Y) / Y.size

        # server
        def gradient_descent(self, X, Y, alpha, iterations):
            """
            Update the model
                Parameters:
                    X (list of ndaray): training data
                    Y (list of ndaray): training labels
                    alpha (float): learning step
                    iterations (int): number of iterations
                Returns:
                    W1 (ndarray): updated first layer weights
                    b1 (ndarray): updated first layer biases
                    W2 (ndarray): updated second layer weights
                    b2 (ndarray): updated second layer biases
            """
            W1, b1, W2, b2 = self.init_params()
            for i in range(iterations):
                Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
                dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
                W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
                if (i % 10 == 0):
                    print("Iteration: ", i)
                    predictions = self.get_predictions(A2)
                    print(self.get_accuracy(predictions, Y))
            return W1, b1, W2, b2

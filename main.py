import numpy as np
import random
import time
import algorithms

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

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        Trains the model and adjust its weights with each iteration
            Parameters:
                training_inputs (ndarray): feature vector of ints
                training_outputs (ndarray): label vector of ints
                training_iterations (int): number of iterations
            Returns:
                model (dict): trained model dictionary
        """
        for _ in range(training_iterations):
            output = self.results(training_inputs)
            error = training_outputs - output
            # Backpropagation: Error weighted derivatives
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_dx(output))
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
        self.weights = self.weights.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output

class Server(Perceptron):
    def __init__(self):
        pass

    def first_model(self):
        """
        Train the first global model to be sent to the devices
            Returns:
                first_model_dict (dict): first model dictionary
        """
        perceptron = Perceptron()

        training_inputs = np.array([[0,0,1],
                                    [1,1,1],
                                    [1,0,1],
                                    [0,0,1],
                                    [0,1,1],
                                    [1,0,1],
                                    [1,0,0],
                                    [0,0,1],
                                    [0,1,1],
                                    [1,1,1]])

        training_outputs = np.array([[0,1,1,0,0,1,1,0,0,1]]).T

        first_model_dict = perceptron.train(training_inputs, training_outputs, 100)
        return first_model_dict

    def aggregation(self, modelList):
        """
        FedAvg weights aggregation function
            Parameters:
                modelList (list): list of models dictionaries
            Returns:
                res (ndarray): averaged aggregated weights of different models
        """
        weights = []
        total = np.zeros(modelList[0].get('weights').shape)
    
        for dict in modelList:
            weights.append(dict['weights'])
        for i in weights:
            total = total + i
           
        res = total / len(modelList)
        return res

    def initialization(self):
        """
        Initialization of local device's features and labels vector
            Returns:
                local_data_list (list): list of inputs
                local_output_list (list): list of labels
                device_list (list): list of devices
        """
        local_data_list = []
        local_output_list = []
        device_list = []

        np.random.seed(1)
        print("Enter the number of devices: ")
        n = int(input())
        print("Configuring local data...: ")
        for i in range(n):
            print("Enter number of rows of matrix (1000 by default): ")
            c = int(input())
            print("Enter number of columns of matrix (3 by default): ")
            d = int(input())

            local_data = np.random.randint(0, 2, (c, d))
            local_output = []
            server = Server()
            devices = Device(i, server.first_model())

            if local_data[i][0] == 0:
                res = 0
                local_out = np.random.randint(0, 2, (c - 1, 1))
                local_out = np.insert(local_output, 0, [res, 0, 0], axis=0)
                local_output.append(local_out)
            else:
                res = 1
                local_out = np.random.randint(0, 2, (c - 1, 1))
                local_out = np.insert(local_output, 0, [res, 0, 0], axis=0)
                local_output.append(local_out)

            local_data_list.append(local_data)
            local_output_list.append(local_output)
            device_list.append(devices)
            
        return local_data_list, local_output_list, device_list

    def model_update(self, weights, local_data_list, local_output_list, device_list, training_iterations):
        """
        Updates the model with new weights and new inputs, labels
            Parameters:
                weights (ndarray): new weights
                local_data_list (list): list of inputs
                local_output_list (list): list of labels
                device_list (list): list of devices
                training_iterations (int): number of training iterations
            Returns:
                trained_model_list (list): list of model dictionaries
        """
        self.weights = weights
        percep = Perceptron()
        trained_model_list = []
        
        for i in range(len(device_list)):
            model_dict = percep.train(local_data_list[i], local_output_list[i], training_iterations)
            device_list[i].first_model_dict = model_dict # here every device gets  the same weights even though different inputs
            trained_model_list.append(model_dict)

        return trained_model_list

class Device(Server):
    def __init__(self, id, first_model_dict):
        self.id = id
        self.first_model_dict = first_model_dict

if __name__ == "__main__":

    # Initialization of objects and local data
    server = Server()
    perceptron = Perceptron()
    local_data_list, local_output_list, device_list = server.initialization()

    # Model updates with weights of first model and local data
    # Or with random weights: perceptron.Weights instead of server.first_model().get('weights')
    trained_model_list = server.model_update(perceptron.weights, local_data_list, local_output_list, device_list, 100)

    print('TRAINED WEIGHTS')
    for i in range(len(device_list)):
        print(trained_model_list[i].get('weights'))

    # Send gradients, aggregation
    agg_weights = server.aggregation(trained_model_list)
    # Send back model updates, updating models
    updated_model_list = server.model_update(agg_weights, local_data_list, local_output_list, device_list, 1000)

    print('AFTER AGGREGATION')
    for i in range(len(device_list)):
        print(updated_model_list[i].get('weights'))

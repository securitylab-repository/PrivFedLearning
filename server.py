import numpy as np
from algorithms import Algorithms
from helpers import Helpers

class Server(Algorithms.Perceptron):

    def __init__(self):
        """
        Server constructor
        """
        pass

    def first_model(self):
        """
        Train the first global model to be sent to the devices
            Returns:
                first_model_dict (dict): first model dictionary
        """
        perceptron = Algorithms.Perceptron()

        X_train = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,0,1],
                            [0,1,1],
                            [1,0,1],
                            [1,0,0],
                            [0,0,1],
                            [0,1,1],
                            [1,1,1]])

        y_train = np.array([[0,1,1,0,0,1,1,0,0,1]]).T

        first_model_dict = perceptron.train(X_train, y_train, 100)
        return first_model_dict

    def initialization(self):
        from device import Device # prevents circular import error
        """
        Initialization of local device's features and labels vector
            Returns:
                local_data_list (list): list of random local inputs, X_train
                local_output_list (list): list of labels, y_train
                device_list (list): list of devices
        """
        local_data_list = []
        local_output_list = []
        device_list = []
        helper = Helpers()

        np.random.seed(1)
        print("Enter the number of devices: ")
        n = int(input())
        print("Configuring local data...: ")
        for i in range(n):
            print("Enter number of rows of matrix (1000 for best performance): ")
            c = int(input())

            # INPUT
            local_data = np.random.randint(0, 2, (c - 1, 3))
            local_output = []
            server = Server()
            devices = Device(i, server.first_model())

            if local_data[i][0] == 0:
                res = 0
                add_res = np.insert(local_data, 0, [res, 0, 0])
                final_input = add_res.reshape(c, 3) 
                local_out = helper.to_dataframe(final_input)['Output'].values
                local_output.append(local_out)

            else:
                res = 1
                add_res = np.insert(local_data, 0, [res, 0, 0])
                final_input = add_res.reshape(c, 3)
                local_out = helper.to_dataframe(final_input)['Output'].values
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
                local_data_list (list): list of local device inputs, X_train
                local_output_list (list): list of labels, y_train
                device_list (list): list of devices
                training_iterations (int): number of training iterations
            Returns:
                trained_model_list (list): list of model dictionaries
        """
        self.weights = weights
        percep = Algorithms.Perceptron()
        trained_model_list = []

        for i in range(len(device_list)):
            model_dict = percep.train(local_data_list[i], local_output_list[i], training_iterations)
            device_list[i].first_model_dict = model_dict
            trained_model_list.append(model_dict)

        return trained_model_list

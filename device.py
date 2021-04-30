import numpy as np
from server import Server
from helpers import Helpers
from algorithms import Algorithms

class Device(Server):
    def __init__(self, id, first_model_dict):
        """
        Device constructor
            Parameters:
                id (int): device identifier
                first_model_dict (dict): dictionary of the server's first model
                which contains: output, error, adjustments, weights
        """
        self.id = id
        self.first_model_dict = first_model_dict

    def initialization(self, list):
        """
        Initialization of local device's features and labels vector
            Returns:
                local_data_list (list): list of random local inputs, X_train
                local_output_list (list): list of labels, y_train
                device_list (list): list of devices
        """
        np.random.seed(1)
        helper = Helpers()
        local_data_list = []
        local_output_list = []
        device_list = []

        print("Configuring local data...: ")
        for i in range(len(list)):
            print("Enter number of rows of matrix for device "  + str(i) + " (1000 for best performance):")
            c = int(input())

            # INPUT
            local_data = np.random.randint(0, 2, (c - 1, 3))
            local_output = []
            server = Server()

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

        return local_data_list, local_output_list

    def local_update(self, weights, local_data_list, local_output_list, device_list, training_iterations):
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

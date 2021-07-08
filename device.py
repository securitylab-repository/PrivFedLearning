import numpy as np
import pandas as pd
from server import Server
from helpers import Helpers
from algorithms import Algorithms
from random import seed
import random


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
            Parameters:
                list (list): list of devices
            Returns:
                local_data_list (list): list of random local inputs, X_train
                local_output_list (list): list of labels, y_train
                c (int): number of observations for the training data
        """
        np.random.seed(1)
        helper = Helpers()
        local_data_list = []
        local_output_list = []

        print("Configuring local data...: ")
        for i in range(len(list)):
            print("Enter number of rows of matrix for device " +
                  str(i) + " (1000 for best performance):")
            c = int(input())

            # INPUT
            local_data = np.random.randint(0, 2, (c - 1, 3))
            local_output = []
            
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
            
        return local_data_list, local_output_list, c

    def random_devices(self):
        """
        Picks random subset of user devices
            Returns:
                rand_devices (list): random selected devices list
                n (int): number of subset of devices
        """
        print('Enter subset max index size (3):')
        n = int(input())
        index_list = list(range(n))
        random.shuffle(index_list)
        rand_devices_list = []
        server = Server()
        device_list = server.create_devices() 
        print('Device list: ')
        print(device_list)

        for i in index_list:
            rand_devices_list.append(device_list[i])

        return rand_devices_list, n

    def random_user_data(self):
        """
        Picks random data for a specific user
            Returns:
                input_list (list): list of random local inputs, X_train
                output_list (list): list of labels, y_train
        """
        server = Server()
        device = Device(0, server.first_model)
        random_device_list, n = self.random_devices()
        print('INSIDE RANDOM USER DATA')
        print('Random device list')
        print(random_device_list)
        input_list, output_list, c = device.initialization(random_device_list)

        outputs_list = []
        for i in range(n):
            new_local_output_list = np.delete(output_list[i][0], [0])
            local_rand_output_list = new_local_output_list.reshape(c-1, 1)
            outputs_list.append(local_rand_output_list)

        return input_list, outputs_list

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
            model_dict = percep.train(
                local_data_list[i], local_output_list[i], training_iterations)
            device_list[i].first_model_dict = model_dict
            trained_model_list.append(model_dict)

        return trained_model_list

    # convert local_data_list, local_output_list to data
    def local_update_nn(self, weights, local_data_list, local_output_list, device_list):
        """
        Updates the model with new weights and new training data with a neural network
            Parameters:
                weights (ndarray): new weights
                local_data_list (list): list of local data
                local_output_list (list): list of local output
                device_list (list): list of devices
            Returns:
                trained_model_list (list): list of model dictionaries
        """
        self.weights = weights
        nn = Algorithms.NeuralNetwork()
        helper = Helpers()
        seed(1)
        df_list = []
        csv_list = []
        data_list = []
        trained_model_list = []
       
        for i in range(len(device_list)):
            df_list.append(pd.DataFrame(np.concatenate([local_data_list[i], local_output_list[i]], axis=1), columns=['Input_0', 'Input_1', 'Input_2', 'Output'])) # added [i] to local_output_list

            for df in df_list:
                for j in range(len(df_list)):
                    df.to_csv(f'df{j}.csv')
                    csv_list.append(f'df{j}.csv')

            data_list.append(helper.df_to_list(csv_list[i]))
            n_inputs = len(data_list[i][0]) - 1
            n_outputs = len(set([row[-1] for row in data_list[i]]))
            network = nn.initialize_network(n_inputs, 2, n_outputs)
            model_dict = nn.train_network(network, data_list[i], 0.5, 10, n_outputs)
            device_list[i].first_model = model_dict
            trained_model_list.append(model_dict)
            
        return trained_model_list

import numpy as np
from algorithms import Algorithms
from helpers import Helpers
from random import seed

class Server(Algorithms):

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
        nn = Algorithms.NeuralNetwork()
        helper = Helpers()
        seed(1)
        data = helper.df_to_list('data.csv')
        n_inputs = len(data[0]) - 1
        n_outputs = len(set([row[-1] for row in data]))
        print('first_model')
        print(n_inputs)
        print(n_outputs)
        network = nn.initialize_network(n_inputs, 2, n_outputs)
        first_model_dict = nn.train_network(network, data, 0.5, 100, n_outputs)
        return first_model_dict

    def create_devices(self):
        """
        Creates a list of devices
            Returns:
                device_list (list): list of devices
        """
        from device import Device
        device_list = []
        server = Server()

        print("Enter the number of devices (users): ")
        n = int(input())
        for i in range(n):
            device_list.append(Device(i, server.first_model))
        return device_list

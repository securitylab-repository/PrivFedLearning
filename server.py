import numpy as np
from algorithms import Algorithms

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

    def create_devices(self):
        """
        Creates a list of devices
            Returns:
                device_list (list): list of devices
        """
        from device import Device
        device_list = []
        server = Server()

        print("Enter the number of devices: ")
        n = int(input())
        for i in range(n):
            device_list.append(Device(i, server.first_model))
        return device_list

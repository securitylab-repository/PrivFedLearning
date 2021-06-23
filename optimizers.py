import numpy as np
import random
from random import randrange
from server import Server
from device import Device

class SGD():
    def __init__(self):
        """
        SGD constructor
        """
        pass

    def random_devices(self):
        """
        Picks random subset of user devices
            Returns:
                rand_devices (list): random selected devices list
        """
        print('Enter subset max index size: (3)')
        n = int(input())
        index_list = list(range(n))
        random.shuffle(index_list)
        rand_devices_list = []
        server = Server()
        device_list = server.create_devices() # 5 for reference, calling create_devices twice bad practice
        print('Device list: ')
        print(device_list)

        for i in index_list:
            rand_devices_list.append(device_list[i])

        return rand_devices_list

    def random_user_data(self):
        """
        Picks random data for a specific user
            Returns:
                input_list (list): list of random local inputs, X_train
                output_list (list): list of labels, y_train
        """
        server = Server()
        device = Device(0, server.first_model)
        random_device_list = self.random_devices()
        print('INSIDE RANDOM USER DATA')
        print('Random device list')
        print(random_device_list)
        input_list, output_list = device.initialization(random_device_list)
        return input_list, output_list

    def loss(self, y, y_hat):
        """
        Returns the MSE loss function
            Parameters:
                y (int): actual output
                y_hat (int): predicted output
        """
        return np.square(y - y_hat).mean()

    def mini_batch_loss_gradient(self, minibatch_size, random_user_minibatch_size, random_user_minibatch, theta):
        """
        Returns the minibatch loss gradient
            Parameters:
                minibatch_size (int): size of virtual minibatch Dt
                random_user_minibatch_size (int): size of randomly selected subset of user's data
                random_user_minibatch (ndarray): random subset of the user's data
                theta (ndarray): parameters
        """
        return 1 / minibatch_size * np.sum(random_user_minibatch_size * self.mini_batch_loss_gradient(minibatch_size, random_user_minibatch_size, random_user_minibatch, theta))

    def mini_batch_sgd(self, alpha, minibatch_size,  random_user_minibatch_size, random_user_minibatch, theta):
        """
        Returns minibatch sgd
            Parameters:
                alpha (float): learning rate
                minibatch_size (int): size of virtual minibatch
                random_user_minibatch_size (int): size of randomly selected subset of user's data
                random_user_minibatch (ndarray): random subset of the user's data
                theta (ndarray): parameters
        """
        theta_next = theta - alpha  * self.mini_batch_loss_gradient(minibatch_size, random_user_minibatch_size, random_user_minibatch, theta)
        return theta_next

    def gradient_descent(self, alpha, minibatch_size,  random_user_minibatch_size, random_user_minibatch, theta):
        """
        Returns a gradient descent step
        Parameters:
            alpha (float): learning rate
            minibatch_size (int): size of virtual minibatch
            random_user_minibatch_size (int): size of randomly selected subset of user's data
            random_user_minibatch (ndarray): random subset of the user's data
            theta (ndarray): parameters
        """
        theta_next = theta - alpha * np.sum(random_user_minibatch_size * self.mini_batch_loss_gradient(minibatch_size, random_user_minibatch_size, random_user_minibatch, theta)) / np.sum(random_user_minibatch_size)
        return theta_next

import numpy as np
import pandas as pd
import math
from algorithms import Algorithms
from server import Server
from aggregations import Aggregations
from helpers import Helpers
from cryptography import Cryptography

if __name__ == "__main__":

    # Initialization of objects and local data
    server = Server()
    agg = Aggregations()
    perceptron = Algorithms.Perceptron()
    nn = Algorithms.NeuralNetwork()
    helper = Helpers()
    device_list = server.create_devices()
    device = device_list[0]
    local_rand_data_list, local_rand_output_list = device.random_user_data() 
    trained_model_list = device.local_update_nn(server.first_model().get('weights'), local_rand_data_list, local_rand_output_list, device_list)

    print('Output of trained_model_list on local data')
    print(trained_model_list[0]['output']) 

    print('TRAINED WEIGHTS')
    for i in range(len(device_list)):
        print(trained_model_list[i].get('weights'))

    # Send gradients, aggregation
    agg_weights = agg.fedAvg(trained_model_list)
    # Send back model updates, updating models
    updated_model_list = device.local_update_nn(agg_weights, local_rand_data_list, local_rand_output_list, device_list)
    print('Output of updated_model_list afer aggregation on local data')
    print(updated_model_list[0]['output'])

    agg_weights_list = agg_weights.tolist()
    print('agg_weights (as a list)')
    print(agg_weights_list)
    agg_weights_str = ''.join(str(e) for e in agg_weights_list)
    print('agg_weights (as strings): ', agg_weights_str)

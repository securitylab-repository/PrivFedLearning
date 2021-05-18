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
    helper = Helpers()

    device_list = server.create_devices()
    device = device_list[0]
    local_data_list, local_output_list = device.initialization(device_list)

    rsa = Cryptography.RSA()
    elg = Cryptography.Elgamal()
    err = Cryptography.LWE(20, [], [], 20, 2, 5, 97)

    # Model updates with weights of first model and local data
    # Or with random weights: perceptron.Weights instead of server.first_model().get('weights')
    trained_model_list = device.local_update(perceptron.weights, local_data_list, local_output_list, device_list, 10000)
    print('Output of trained_model_list on local data')
    print(trained_model_list[0]['output'])

    # Send gradients, aggregation
    agg_weights = agg.fedAvg(trained_model_list)
    # Send back model updates, updating models
    updated_model_list = device.local_update(agg_weights, local_data_list, local_output_list, device_list, 10000)
    print('Output of updated_model_list afer aggregation on local data')
    print(updated_model_list[0]['output'])

    print('**************** RSA ****************')
    print('Original message')
    print(agg_weights)
    agg_weights_str = helper.ndarray_to_string(agg_weights)
    enc, dec = rsa.init_rsa(32, agg_weights_str)
    print('Encrypted message')
    print(enc)
    print('Decrypted message')
    print(dec)

    print('\n**************** El Gamal ****************')
    print('Original message')
    print(agg_weights)
    agg_weights_float = helper.ndarray_to_int(agg_weights)
    elg.elgamal(agg_weights, 14867) # pb because b should be an int

    elg.elgamal(300, 14867)

    print('\n**************** LWE ****************')
    err.lwe() # works if M is one bit (0 or 1)

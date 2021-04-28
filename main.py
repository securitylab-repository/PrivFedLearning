import numpy as np
import pandas as pd
from algorithms import Algorithms
from server import Server
from device import Device
from aggregations import Aggregations
from helpers import Helpers

if __name__ == "__main__":

    # Initialization of objects and local data
    server = Server()
    agg = Aggregations()
    perceptron = Algorithms.Perceptron()
    helper = Helpers()
    local_data_list, local_output_list, device_list = server.initialization()
    df = helper.to_dataframe(local_data_list[0])
    df.to_csv("data.csv", sep=',', index=False)
    df = pd.read_csv('data.csv', delimiter=',')

    # Model updates with weights of first model and local data
    trained_model_list = server.model_update(perceptron.weights, local_data_list, local_output_list, device_list, 10000)

    # Send gradients, aggregation
    agg_weights = agg.fedAvg(trained_model_list)
    # Send back model updates, updating models
    updated_model_list = server.model_update(agg_weights, local_data_list, local_output_list, device_list, 10000)

    # TEST DATA
    X_test = np.array([[1,0,1],
                        [0,1,1],
                        [1,0,1],
                        [0,1,1]])

    y_test = np.array([[1,0,1,0]]).T

    perceptron.train_test(X_test, y_test, 100000)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New test data: ", A, B, C)
    print("Predicted: ")
    print(perceptron.results(np.array([A, B, C])))

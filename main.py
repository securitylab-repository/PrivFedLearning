import numpy as np

class Perceptron():
    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Set weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.weights = 2 * np.random.random((3, 1)) - 1

    '''def exp_test(self, x):
        x.astype(float)
        return np.exp(-x)'''

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        '''print('*************INSIDE SIGMOID**************')
        print(type(x))
        print(x.shape)
        print(x)
        y = 1 / (1 + np.exp(-x))
        print('RESULT OF SIGMOID')
        print(type(y))
        print(y.shape)
        print(y)'''
        return 1 / (1 + np.exp(-x))

    def sigmoid_dx(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        '''print('***************INSIDE SIGMOID_DX***************')
        print(type(x))
        print(x.shape)
        print(x)
        y = x*(1-x)
        print('RESULT OF SIGMOID_DX')
        print(type(y))
        print(y.shape)
        print(y)'''
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model through trial and error, adjusting the
        weights each time to get a better result
        """

        for _ in range(training_iterations):
            # Pass training set through the neural network
            output = self.results(training_inputs)
            # Calculate the error rate
            error = training_outputs - output
            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_dx(output))
            # Adjust weights
            self.weights = self.weights + adjustments

        final_weights = self.weights

        model = {'output':output,
                 'error':error,
                 'adjustments':adjustments,
                 'weights':self.weights,
                 'final_weights':final_weights
                }

        return model

    def results(self, inputs):
        """
        Pass inputs through the neural network to get output
        """
        inputs = inputs.astype(float)
        self.weights = self.weights.astype(float)
        '''print("Input type: ")
        print(type(inputs))
        print("Weights type: ")
        print(type(self.weights))'''
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output

    '''def get_model_dict(self, model_dict, key):
        pass'''

class Server(Perceptron):
    def __init__(self):
        pass

    # Train the first global model
    def first_model(self):
        perceptron = Perceptron()

        training_inputs = np.array([[0,0,1],
                                    [1,1,1],
                                    [1,0,1],
                                    [0,1,1]])

        # label
        training_outputs = np.array([[0,1,1,0]]).T

        first_model_dict = perceptron.train(training_inputs, training_outputs, 10)
        return first_model_dict

    def aggregation(self, modelList):
        weights = []
        sum = 0.0

        print('**********INSIDE AGGREGATATION*********')
        print('Model List')
        print(type(modelList))
        print(modelList)
        for dict in modelList:
            for i in dict:
                weights.append(modelList[dict][i].get('final_weights'))
        for i in weights:
            sum += i

        res = sum / len(modelList)
        return res

    def initialization(self):
        local_data_list = []
        local_output_list = []
        device_list = []
        #model_list = []
        np.random.seed(1)
        print("Enter the number of devices: ")
        n = int(input())
        print("Configuring local data...: ")
        for i in range(n):
            print("Enter lower number (0 by default): ")
            a = int(input())
            print("Enter limit number (2 by default): ")
            b = int(input())
            print("Enter number of rows of matrix (1000 by default): ")
            c = int(input())
            print("Enter number of columns of matrix (3 by default): ")
            d = int(input())

            local_data = np.random.randint(a, b, (c, d))
            print('*****************INSIDE INITIALIZATION****************')
            print(type(local_data))
            print(local_data.shape)
            print(local_data)
            local_output = []
            server = Server()
            devices = Device(i, server.first_model())


            if local_data[i][0] == 0:
                res = 0
                #local_output.append(res)
                local_out = np.random.randint(a, b, (c - 1, 1))
                local_out = np.insert(local_output, 0, [res, 0, 0], axis=0)
                local_output.append(local_out)
            else:
                res = 1
                #local_output.append(res)
                local_out = np.random.randint(a, b, (c - 1, 1))
                local_out = np.insert(local_output, 0, [res, 0, 0], axis=0)
                local_output.append(local_out)

            local_data_list.append(local_data)
            local_output_list.append(local_output)
            device_list.append(devices)

        print('LOCAL OUTPUT')
        print(type(local_output))
        #print(local_output.shape)
        print(local_output)
        #local_output_list = np.asarray(local_output_list, dtype='object')
        return local_data_list, local_output_list, device_list

    def model_update(self, weights, local_data_list, local_output_list, device_list, training_iterations):
        self.weights = weights
        percep = Perceptron()
        trained_model_list = []

        local_output_list = np.asarray(local_output_list, dtype='object')
        print('*************INSIDE MODEL UPDATE METHOD*******************')
        print(type(local_data_list), type(local_output_list), type(device_list)) # 3 class list
        print('input data: ', type(local_data_list[0]), 'output data: ', type(local_output_list[0])) # LIST != NDARRAY
        print(local_data_list[0].shape, local_output_list[0].shape)

        for i in range(len(local_data_list)):
            model_dict = percep.train(local_data_list[i], local_output_list[i], training_iterations)
            device_list[i].first_model_dict = model_dict
            trained_model_list.append(model_dict)
        return trained_model_list

class Device(Server):
    def __init__(self, id, first_model_dict):
        self.id = id
        self.first_model_dict = first_model_dict

    '''def agg_update(self, training_inputs, training_outputs, training_iterations, modelList):
        serv = Server()
        self.weights = serv.aggregation(modelList)
        model_update(self.weights, training_inputs, training_outputs, training_iterations)'''

class FederatedAvg():
    pass
# Separer les classes dans des fichiers a part
if __name__ == "__main__":

    server = Server()
    #initializer = Device(10, server.first_model())
    local_data_list, local_output_list, device_list = server.initialization()
    # Model updates with weights of first model and local initialization data
    #trained_model_list = initializer.model_update(initializer.first_model_dict.get('final_weights'), local_data_list, local_output_list, device_list, 10000)
    trained_model_list = server.model_update(server.first_model().get('final_weights'), local_data_list, local_output_list, device_list, 10)
    # Send gradients
    # Aggregation
    agg_weights = server.aggregation(trained_model_list)
    # Send back model updates
    # Updating models
    #updated_model_list = initializer.model_update(agg_weights, local_data_list, local_output_list, device_list, 10000)

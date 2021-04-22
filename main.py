import numpy as np

class Perceptron():
    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Set weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_dx(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model abd adjusting the weights 
        """
        for _ in range(training_iterations):
            output = self.results(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_dx(output))
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
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output

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
            local_output = []
            server = Server()
            devices = Device(i, server.first_model())


            if local_data[i][0] == 0:
                res = 0
                local_out = np.random.randint(a, b, (c - 1, 1))
                local_out = np.insert(local_output, 0, [res, 0, 0], axis=0)
                local_output.append(local_out)
            else:
                res = 1
                local_out = np.random.randint(a, b, (c - 1, 1))
                local_out = np.insert(local_output, 0, [res, 0, 0], axis=0)
                local_output.append(local_out)

            local_data_list.append(local_data)
            local_output_list.append(local_output)
            device_list.append(devices)

        return local_data_list, local_output_list, device_list

    def model_update(self, weights, local_data_list, local_output_list, device_list, training_iterations):
        self.weights = weights
        percep = Perceptron()
        trained_model_list = []
        local_output_list = np.asarray(local_output_list, dtype='object')

        for i in range(len(local_data_list)):
            model_dict = percep.train(local_data_list[i], local_output_list[i], training_iterations)
            device_list[i].first_model_dict = model_dict
            trained_model_list.append(model_dict)
        return trained_model_list

class Device(Server):
    def __init__(self, id, first_model_dict):
        self.id = id
        self.first_model_dict = first_model_dict

if __name__ == "__main__":

    server = Server()
    local_data_list, local_output_list, device_list = server.initialization()
    trained_model_list = server.model_update(server.first_model().get('final_weights'), local_data_list, local_output_list, device_list, 10)
    # Send gradients
    # Aggregation
    agg_weights = server.aggregation(trained_model_list)
    # Send back model updates
    # Updating models
    #updated_model_list = initializer.model_update(agg_weights, local_data_list, local_output_list, device_list, 10000)

import numpy as np
import pandas as pd
from optimizers import SGD
import tensorflow as tf
from sklearn.model_selection import train_test_split
from algorithms import Algorithms

# ------------------ Random devices ---------------------
sgd = SGD()
'''random_device_list = sgd.random_devices()
print('Random device list:')
print(random_device_list)'''

# 940, 6D0, 160, 1C0, 220
# 160, 940, 6D0
# ------------------ TESTS: OK --------------------------

# ------------------ Random user data -------------------
random_user_data_list, random_user_output_list = sgd.random_user_data()
random_user_data = random_user_data_list[0] # should pick randomly one or several data from random_user_data_list instead of hard coding, minibatch_size = len(random_user_data)
random_user_output = random_user_output_list[0]

print('Random user data: ')
print(type(random_user_data))
print(random_user_data)
print('Random user output: ')
random_user_output = np.asarray(random_user_output)
print(type(random_user_output))
print(random_user_output) # first element doesn't count

# 910, 130, 160
# ------------------ TESTS: OK --------------------------

# ------------------ Loss function with Keras----------------------
df = pd.read_csv('data.csv')
x = df.loc[:, df.columns != 'Output'].to_numpy()
y = df.loc[:, df.columns == 'Output'].to_numpy()
'''print(type(x), type(y)) # DataFrame
print(x)
print(y)'''


df['Input'] = df[df.columns[0:-1]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1) # converting Input_0 etc. into Input
df['Input'] = pd.to_numeric(df['Input']) # not used
#df['Input'].to_csv("x.csv", sep=',', index=False)
#print('Input type', df.dtypes)
#df['Input'] = df[['Input_0', 'Input_1', 'Input_2']].astype(str).agg(''.join, axis=1).astype(int)
#print('INPUT')
#print(df['Input'])
#print(df)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.20,random_state=42)
X_Train = X_Train.astype('int64')
X_Test = X_Test.astype('int64')
#print(X_Train.shape, Y_Train.shape, X_Test.shape, Y_Test.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Dense(3))
model.compile(optimizer='sgd', loss='mse')
model.fit(X_Train, Y_Train, batch_size=10, epochs=100)
y_pred = model.predict(X_Train)
'''print('y_pred', y_pred)
print('y_train', Y_Train)'''

print('Numpy')
se = (Y_Train - y_pred)**2
mse = np.mean(se)
print(mse)
print('SGD MSE')
print(sgd.loss(Y_Train, y_pred))
# ------------------ TESTS: OK --------------------------

# --------------- Gradient Descent with own NN --------------
minibatch_size = len(random_user_data_list)
random_user_minibatch_size = len(random_user_data) # len of ndarray != list
random_user_minibatch = random_user_data
theta = X_Train
#mblg = sgd.mini_batch_loss_gradient(minibatch_size, random_user_minibatch_size, random_user_minibatch, theta)
nn = Algorithms.NeuralNetwork()
W1, b1, W2, b2 = nn.gradient_descent(random_user_data, random_user_output, 0.1, 500)
# ------------------ TESTS: Errors ----------------------











'''myndarray = np.asarray([[ 5.98588102,  5.98588102, -5.98669299,  5.98588102, -5.98669299, -5.98669299,
  -5.98669299, -5.98669299,  5.98588102,  5.98588102, -5.98669299, -5.98669299],
 [ 5.63582643,  5.63582643, -5.64392147,  5.63582643, -5.64392147, -5.64392147,
  -5.64392147, -5.64392147,  5.63582643,  5.63582643, -5.64392147, -5.64392147],
 [ 5.84532688,  5.84532688, -5.84088786,  5.84532688, -5.84088786, -5.84088786,
  -5.84088786, -5.84088786,  5.84532688,  5.84532688, -5.84088786, -5.84088786]])

print(type(myndarray))
print(myndarray.shape)
print(myndarray)

ndarray_list = myndarray.tolist()
print(type(ndarray_list))
print(ndarray_list)

ndarray_str = ''.join(str(e) for e in ndarray_list)
print(type(ndarray_str))
print(ndarray_str)
print(len(ndarray_str))

ndarray_ascii = [ord(i) for i in ndarray_str]
print(type(ndarray_ascii))
print(ndarray_ascii)
print(len(ndarray_ascii))

ndarray_int = int(''.join(str(ord(c)) for c in ndarray_str[:2]))
print(ndarray_int)'''


'''ndarray_float = [float(i) for i in ndarray_list]
print(ndarray_float)'''

'''class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def think(self, inputs):
        """
        Pass inputs through the neural network to get output
        """

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 100000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))'''

'''if __name__ == "__main__":
    np.random.seed(int(time.time()))
    a = []
    a.append(1)
    a.append(np.random.randint(0, 2, (5, 3)))

    print(type(a))
    print(a)

    a = np.concatenate([[1], a])
    print(type(a))
    print(a.shape)
    print(a)

    b = np.random.randint(0, 2, (5, 3))
    print(type(b))
    print(b.shape)
    print(b)

    b = np.append([1, 0, 0], b)
    print(type(b))
    print(b.shape)
    print(b)

    b = np.insert(b, 0, [1, 0, 0])
    print(type(b))
    print(b.shape)
    print(b)

    c = np.insert(b, 0, [1, 0, 0], axis=0)
    print(type(c))
    print(c.shape)
    print(c)

    d = np.arange(18).reshape(6, 3)
    print(d)
    e = c + d
    print(e)

    f = e / 2
    print(f)

    print('Random values from -1 to 1')
    g = 2 * np.random.random((3, 1)) - 1
    print(g)

    print('Initial matrix')
    h = np.ones((5, 3))
    print(h)

    print('Zeros')
    i = np.zeros((5, 3))

    print('Sum')
    print(h + i)

    print('Average')
    print((h + i) / 2)

    j = np.array([[1, 0, 1],
                 [0, 0, 1],
                 [1, 1, 0]])
    print(j.shape)

    print('WEIGHTS TYPE')
    weights = 2 * np.random.random((3, 1)) - 1
    print(type(weights))
    print(weights)'''

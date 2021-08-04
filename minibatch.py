import numpy as np


import algorithm as algorithm

def sigmoid( x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))



def sigmoid_deriv(x):
    
    # compute the derivative of the sigmoid function ASSUMING # that ‘x‘ has already been passed through the ‘sigmoid‘ # function
    return x * (1-x)



class NeuralNetwork(algorithm.TAlgorithm):

    def __init__(self, layers : list,  activation_function=sigmoid, activation_derivation=sigmoid_deriv) -> None:
        # initialize the list of weights matrices, then store the network architecture and learning rate
        self.W = []
        self.layers = layers
        self.actiovation_function = activation_function
        self.activation_derivation = activation_derivation
        # start looping from the index of the first layer but
        # stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
             # randomly initialize a weight matrix connecting the
             # number of nodes in each respective layer together,
             # adding an extra node for the bias
             w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
             self.W.append(w / np.sqrt(layers[i]))
             
        
        # the last two layers are a special case where the input
        # connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    
    def set_parameter(self,W):
        self.W = W

    def get_parameter(self):

        return self.W
       

    def __repr__(self):
        # construct and return a string that represents the network
        # # architecture
        print("\n")
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))



    def fit(self,X_train_batch, y_train_batch) :
        
        nabla_W = [np.zeros_like(W) for W in self.W]
        
        for x, y in zip(X_train_batch, y_train_batch):
            
            
            delta_nabla_W = self.backpropagation(x, y)
            nabla_W = [dnW + nW for nW, dnW in zip(delta_nabla_W, nabla_W)]

        return nabla_W,len(X_train_batch) 
        
    
    def backpropagation(self, x, y):
        # construct our list of output activations for each layer
        # as our data point flows through the network; the first
        # activation is a special case -- it’s just the input
        # feature vector itself
        A = [np.atleast_2d(x)]
        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])
       
            # computing the "net output" is simply applying our
            # nonlinear activation function to the net input
            out = self.actiovation_function(net)
       
            # once we have the net output, add it to our list of
            # activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activations list) and the true target
        # value
        error = A[-1] - y 
        # from here, we need to apply the chain rule and build our
        # list of deltas ‘D‘; the first entry in the deltas is
        # simply the error of the output layer times the derivative
        # of our activation function for the output value
        D = [error * self.activation_derivation(A[-1])]

        # once you understand the chain rule it becomes super easy
        # to implement with a ‘for‘ loop -- simply loop over the
        # layers in reverse order (ignoring the last two since we
        # already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.activation_derivation(A[layer])
            D.append(delta)

        # since we looped over our layers in reverse order we need to reverse the deltas
        D= D[::-1]
        for layer in np.arange(0, len(self.W)):
            # make the dot product of the layer activations with their respective deltas
            D[layer] =  A[layer].T.dot(D[layer])

        return D
        
    


    def predict(self, X, addBias=True):
        # initialize the output prediction as the input features -- this
        # value will be (forward) propagated through the network to
        # obtain the final prediction
        p = np.atleast_2d(X)

        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1’s as the last entry in the feature
            # matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking
            # the dot product between the current activation value ‘p‘
            # and the weight matrix associated with the current layer,
            # then passing this value through a nonlinear activation
            # function
            p= self.actiovation_function(np.dot(p, self.W[layer]))

        # return the predicted value
        return p



    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        
        # return the loss
        return loss

    

#from sklearn.preprocessing import LabelBinarizer
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
#from sklearn import datasets
#
##construct the XOR dataset
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [0]])
##define our 2-2-1 neural network and train it
#nn = NeuralNetwork([2, 2,1], alpha=0.5)
#nn.fit(X, y, epochs=20000)
##
#for (x, target) in zip(X, y):
#    # make a prediction on the data point and display the result
#    # to our console
#    pred = nn.predict(x)[0][0]
#    step= 1 if pred > 0.5 else 0
#    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))
#
#
## load the MNIST dataset and apply min/max scaling to scale the
## pixel intensity values to the range [0, 1] (each image is
## represented by an 8 x 8 = 64-dim feature vector)
#print("[INFO] loading MNIST (sample) dataset...")
#digits = datasets.load_digits()
#data = digits.data.astype("float")
#data = (data - data.min()) / (data.max() - data.min())
#print("[INFO] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))
##
##
### construct the training and testing splits
#(trainX, testX, trainY, testY) = train_test_split(data,digits.target, test_size=0.25)
### convert the labels from integers to vectors
#trainY = LabelBinarizer().fit_transform(trainY)
#testY = LabelBinarizer().fit_transform(testY)
##
##
### train the network
#print("[INFO] training network...")
#nn = NeuralNetwork([trainX.shape[1], 512, 10])
#print("[INFO] {}".format(nn))
#nn.fit(trainX, trainY, epochs=1000)
##
### evaluate the network
#print("[INFO] evaluating network...")
#predictions = nn.predict(testX)
#predictions = predictions.argmax(axis=1)
#print(classification_report(testY.argmax(axis=1), predictions))
#
#



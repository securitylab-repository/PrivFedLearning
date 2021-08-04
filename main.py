
import numpy as np
import nndevice as nd
import nnserver as ns
import minibatch as minibatch_nn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

#construct the XOR dataset
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [0]])

## load the MNIST dataset and apply min/max scaling to scale the
## pixel intensity values to the range [0, 1] (each image is
## represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))

### construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data,digits.target, test_size=0.25)
### convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

#define algo
algo = minibatch_nn.NeuralNetwork([trainX.shape[1], 256, 10])

### initialize a server 
server = ns.NNServer(algo)

server.init_training(trainX,trainY,batch_size=256)

server.start_training_parallel(epochs=10000)


## evaluate the network
print("[INFO] evaluating network...")
predictions = server.algo.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))



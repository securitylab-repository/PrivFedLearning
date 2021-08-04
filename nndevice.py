

import numpy as np
import algorithm as algorithm

class NNDevice:

    device_id = 0 

    def __init__(self, X : np.ndarray , y : np.ndarray, batch_size : int ) -> None:

        NNDevice.device_id+=1
        self.id = NNDevice.device_id
        self.X = X
        self.y = y 
        self.batch_size = batch_size

        #self.trainX = np.c_[self.trainX, np.ones((self.trainX.shape[0]))]

    def set_algo(self,algo : algorithm.TAlgorithm) -> None:

        self.algo = algo


    def __repr__(self) -> str:

        return "Device{}: Training Algorithm is {}".format(self.id,self.algo)


    def update_parameter(self, parameter):
        self.algo.set_parameter(parameter) 

    def fit(self):
        
        idx = np.random.permutation(self.batch_size)
        shuffled_X, shuffled_y = self.X[idx], self.y[idx]
        return self.algo.fit(shuffled_X,shuffled_y)

        
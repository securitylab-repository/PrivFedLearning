
import numpy as np
import nndevice as nnd
import algorithm as algorithm
class NNServer() :

    def __init__(self,algo : algorithm.TAlgorithm ) -> None:
        """Init the devices list and set the training algotithm"""
       
        self.devices = []
        self.algo = algo

    def add_device(self,device : nnd.NNDevice) -> None:
        """Add a signle device to the server devices by setting its algo training"""
        
        device.set_algo(self.algo)
        self.devices.append(device)

    def add_devices(self, devices : list) -> None:
        """Add a list of devices to the server"""
        for d in devices :
            self.add_device(d)
    
    def init_training(self, X : np.ndarray, y : np.ndarray, batch_size=128):
        """ Init the training.
            X : the training set
            y : the training target
            batch_size : the soze of the batch"""

        # init list of devices 
        self.devices = []

        # add one column to the dataset
        X = np.c_[X, np.ones((X.shape[0]))]

        self.X = X
        self.y = y
        
        N = len(y)
        batches = [zip(X[b:b+batch_size], y[b:b+batch_size]) for b in range(0, N, batch_size)]
        
        for i in range(len(batches)-1):

            device = nnd.NNDevice(X,y, batch_size )
            self.add_device(device)
    


    def start_training(self, epochs=1000, displayUpdate=100, alpha=0.1) -> None:

        if  not self.devices :
            return
         
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):

            for device in self.devices:
                device.update_parameter(self.algo.get_parameter())
                gradient, len_batch = device.fit()
                # Update parameters W
                self.algo.set_parameter([W - (alpha * nW / len_batch) for W, nW in zip(self.algo.get_parameter(), gradient)])
            
            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.algo.calculate_loss(self.X, self.y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    
    def start_training_parallel(self, epochs=1000, displayUpdate=100, alpha=0.1) -> None:

        if  not self.devices :
            return
         
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            
            gradients = []
            initial = True
            for device in self.devices:

                if (initial):
                    gradient, len_batch = device.fit()
                else:
                    device.update_parameter(self.algo.get_parameter())
                    gradient, len_batch = device.fit()
                if not gradients:
                    #print("Server: initial gradient")
                    gradients =  [ nW / len_batch for nW in gradient]
                    #print(gradients)
                else:
                    #print("Server: aggregate gradient")
                    gradients = [ (g + nW / len_batch) / 2 for g, nW in zip(gradients,gradient)]
                    #print(gradients)
            
            # Update parameters W
            #print("Server : update parameter")
            self.algo.set_parameter([W - (alpha * g) for W, g in zip(self.algo.get_parameter(), gradients)])
            initial = False
            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.algo.calculate_loss(self.X, self.y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
                

    
    def __repr__(self) -> str:
        
        return "Server: Training algorithm is {}".format(self.algo)


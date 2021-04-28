import numpy as np

class Aggregations():

    def __init__(self):
        """
        Aggregations constructor
        """
        pass

    def fedAvg(self, modelList):
        """
        FedAvg weights aggregation function
            Parameters:
                modelList (list): list of models dictionaries
            Returns:
                res (ndarray): averaged aggregated weights of different models
        """
        weights = []
        sum = np.zeros(modelList[0].get('weights').shape)

        for dict in modelList:
            weights.append(dict['weights'])

        for i in weights:
            sum = sum + i

        res = sum / len(modelList)
        return res

    def fedProx(self):
        """
        fedProx averaging 
        """
        pass

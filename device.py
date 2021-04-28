from server import Server

class Device(Server):
    def __init__(self, id, first_model_dict):
        """
        Device constructor
            Parameters:
                id (int): device identifier
                first_model_dict (dict): dictionary of the server's first model 
        """
        self.id = id
        self.first_model_dict = first_model_dict

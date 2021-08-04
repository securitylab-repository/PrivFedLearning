import abc
class TAlgorithm(metaclass=abc.ABCMeta) :

    @abc.abstractmethod
    def fit(self):
        pass
    
    @abc.abstractmethod
    def get_parameter(self):

        pass
    
    @abc.abstractmethod
    def set_parameter(self):

        pass

    @abc.abstractmethod
    def calculate_loss(self) :

        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is TAlgorithm:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented

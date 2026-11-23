from abc import ABC,abstractmethod
class BaseModel(ABC):
    def __init__(self,name) -> None:
        self.name=name
        self.model=None
        self.metrics=None        
    @abstractmethod
    def build(self):
        pass
    @abstractmethod
    def train(self,X_train,y_train):
        pass
    @abstractmethod
    def evaluate(self,X_test,y_test):
        pass


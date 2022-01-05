import numpy as np

class LogisticRegression:
    
    def __init__(self) -> None:
        pass    
    
    
    def sigmoid(self, x):
       
       return 1.0/(1+np.exp(-x))
        
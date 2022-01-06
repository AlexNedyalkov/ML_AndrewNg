import numpy as np

class LogisticRegression:
    
    def __init__(self, lr = 0.01, n_iters = 500) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.bias = None
        self.weights = None    
    
    
    def sigmoid(self, x):
       
       return 1.0/(1+np.exp(-x))
   
    def fit(self, X, y):
        n_samples, m_features = X.shape
        X = X.values
        y = y.values
        X = np.hstack((np.ones((X.shape[0], 1)), X)) 

       
        # initiate the model 
        self.weights = np.zeros(m_features)
        self.bias = 0
        
       
       # gradient descent
    #    delta_w = 1/n_samples * ()
    #    delta_b = 1/samples * ()
       
    #    self.w -= delta_w
    #    self.b -= delta_b
       
        
    def predict(self, X):
        
        linear_model = np.dot(X, self.weights) + self.b    
        return np.array([0 if x < 0.5 else 1 for x in self.sigmoid(linear_model)])   
    
        
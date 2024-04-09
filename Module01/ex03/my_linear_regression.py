import numpy as np

class MyLinearRegression(): 
    """
    Description:
        My personnal linear regression class to fit like a boss.
        • fit_(self, x, y),
        • predict_(self, x),
        • loss_elem_(self, y, y_hat),
        • loss_(self, y, y_hat).
        • mse_(self, y, y_hat):
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000): 
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def add_intercept(self, x):
        if x.ndim == 1:
            x = x.reshape(x.size, 1)
        one = np.ones((x.shape[0], 1))
        return np.concatenate((one, x), axis=1)

    def predict_(self, x):
        X = self.add_intercept(x)
        y_hat = np.dot(X, self.thetas)
        return y_hat

    def simple_gradient(self, x, y):
        m = x.shape[0]
        modified_x = np.hstack([np.ones((m, 1)), x])
        transpose_x = modified_x.T
        expected_y = np.dot(modified_x, self.thetas)
        j = np.dot(transpose_x, expected_y - y) / m
        return j

    def fit_(self, x, y): 
        if not isinstance(x, np.ndarray) or x.size == 0 or \
            not isinstance(y, np.ndarray) or y.size == 0 or \
            x.shape[0] != y.shape[0]:
            return None
        
        new_theta = self.thetas.astype('float64').copy()
        for _ in range(self.max_iter):
            gradient = self.simple_gradient(x, y)
            new_theta -= self.alpha * gradient
            self.thetas = new_theta
    
    def loss_elem_(self, y, y_hat):
        return (y_hat - y) ** 2
    
    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.size == 0 or \
            not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
            y.ndim != y_hat.ndim:
            return None
        cost = np.sum(self.loss_elem_(y, y_hat)) / (2 * len(y))
        return cost
    
    def mse_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.size == 0 or \
            not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
            y.size != y_hat.size:
            return None
        
        mse = np.sum((y_hat - y) ** 2) / y.size
        return mse
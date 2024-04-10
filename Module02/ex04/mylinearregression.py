import numpy as np

def print_array(x):
    print("array(", end="")
    for idx, item in enumerate(x):
        if idx < len(x) - 1:
            print(f"{item}, ", end="")
        else:
            print(f"{item}", end="")
    print(")")

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
        if not isinstance(alpha, float) or \
            not isinstance(max_iter, int): 
            raise ValueError("Invalid input values")
        
        thetas = np.array(thetas)
        if thetas.size == 0 or not thetas.ndim == 2:
            raise ValueError("thetas must be a non-empty 2D numpy.ndarray")
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = thetas

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.size == 0 or x.ndim != 2 or \
                x.shape[1] + 1 != self.theta.shape[0]:
            return None
        
        x_modified = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = np.dot(x_modified, self.theta)
        print_array(y_hat)
        return y_hat

    def gradient(self, x, y):   
        m = x.shape[0]
        modified_x = np.hstack([np.ones((m, 1)), x])
        transpose_x = modified_x.T    
        expected_y = np.dot(modified_x, self.theta)
        disparity = expected_y - y
        j = np.dot(transpose_x, disparity) / m
        return j

    def fit_(self, x, y): 
        if not isinstance(x, np.ndarray) or x.size == 0 or x.ndim != 2 or\
            not isinstance(y, np.ndarray) or y.size == 0 or \
            x.shape[0] != y.shape[0]:
            return None
        if x.shape[1] + 1 != self.theta.shape[0]:
            self.theta = np.insert(self.theta, 0, 0, axis=0)

        for _ in range(self.max_iter):
            new_gradient = self.gradient(x, y)
            self.theta -= self.alpha * new_gradient

        print_array(self.theta )
        return self.theta 
    
    def loss_elem_(self, y, y_hat):
        res = (y_hat - y) ** 2
        print_array(res)
        return res
    
    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.size == 0 or \
            not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
            y.ndim != y_hat.ndim:
            return None
        cost = np.sum(self.loss_elem_(y, y_hat)) / (2 * len(y))
        print(cost)
        return cost

if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]]) 
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
    # Example 0:
    y_hat = mylr.predict_(X)
    # Output:
    # array([[8.], [48.], [323.]])
    # Example 1:
    mylr.loss_elem_(Y, y_hat)
    # Output:
    # array([[225.], [0.], [11025.]])
    # Example 2:
    mylr.loss_(Y, y_hat)
    # Output:
    # 1875.0
    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    # Output:
    # array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
    # Example 4:
    y_hat = mylr.predict_(X)
    # Output:
    # array([[23.417..], [47.489..], [218.065...]])
    # Example 5:
    mylr.loss_elem_(Y, y_hat)
    # Output:
    # array([[0.174..], [0.260..], [0.004..]])
    # Example 6:
    mylr.loss_(Y, y_hat)
    # Output:
    # 0.0732..
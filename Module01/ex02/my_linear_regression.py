import numpy as np

class MyLinearRegression(): 
    """
    Description:
        My personnal linear regression class to fit like a boss.
        • fit_(self, x, y),
        • predict_(self, x),
        • loss_elem_(self, y, y_hat),
        • loss_(self, y, y_hat).
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
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """

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
        """Computes the half mean squared error of two non-empty numpy.array, without any for loop. 
        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
        """
        if not isinstance(y, np.ndarray) or y.size == 0 or \
            not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
            y.ndim != y_hat.ndim:
            return None
        cost = np.sum(self.loss_elem_(y, y_hat)) / (2 * len(y))
        return cost

if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554] ])

    lr1 = MyLinearRegression(np.array([[2], [0.7]]))

    # Example 0.0:
    y_hat = lr1.predict_(x)
    print(y_hat)
    # # Output:
    # array([[10.74695094],
    #     [17.05055804],
    #     [24.08691674],
    #     [36.24020866],
    #     [42.25621131]])

    # Example 0.1:
    print(lr1.loss_elem_(y, y_hat))
    # # Output:
    # array([[710.45867381],
    # [364.68645485],
    # [469.96221651],
    # [108.97553412],
    # [299.37111101]])
    # Example 0.2:
    print(lr1.loss_(y, y_hat))
    # # Output:
    # 195.34539903032385
    # # Example 1.0:
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print(lr2.thetas)
    # # Output:
    # array([[1.40709365],
    # [1.1150909 ]])
    # Example 1.1:
    y_hat = lr2.predict_(x)
    print(y_hat)
    # # Output:
    # array([[15.3408728 ],
    # [25.38243697],
    # [36.59126492],
    # [55.95130097],
    # [65.53471499]])
    # Example 1.2:
    print(lr2.loss_elem_(y, y_hat))
    # # Output:
    # array([[486.66604863],
    # [115.88278416],
    # [ 84.16711596],
    # [ 85.96919719],
    # [ 35.71448348]])
    # Example 1.3:
    print(lr2.loss_(y, y_hat))
    # # Output:
    # 80.83996294128525

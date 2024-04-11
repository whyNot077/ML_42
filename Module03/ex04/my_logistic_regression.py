import numpy as np
import math

class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def sigmoid_(self, x):
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        return np.array(1 / (1 + np.exp(-x)))

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        x = check_valid(x)
        if x is None:
            return None
        self.theta = check_valid_theta(self.theta, x.shape[1])
        if self.theta is None:
            return None
        modified_x = np.hstack((np.ones((x.shape[0], 1)), x))
        return np.array(1 / (1 + np.exp(-modified_x.dot(self.theta))))

    def loss_(self, y, y_hat, eps=1e-15):
        """
        Computes the logistic loss value.
        Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
        Returns:
        The logistic loss value as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        v = [y, y_hat]
        for i in range(len(v)):
            v[i] = check_valid_1(v[i])
            if v[i] is None:
                return None
        y, y_hat = v

        if y.shape != y_hat.shape or not isinstance(eps, float):
            return None
        
        y_hat = np.clip(y_hat, eps, 1 - eps)  
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return float(loss)

    def gradient_(self, x, y):
        """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible shapes.
        Raises:
        This function should not raise any Exception.
        """
        x = check_valid(x)
        y = check_valid_1(y)
        if x is None or y is None:
            return None
        self.theta = check_valid_theta(self.theta, x.shape[1])
        if self.theta is None:
            return None
        
        x_with_bias = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = self.sigmoid_(np.dot(x_with_bias, self.theta))
        gradient = np.dot(x_with_bias.T, y_hat - y) / y.size
        return gradient

    def fit_(self, x, y):
        for _ in range(self.max_iter):
            new_gradient = self.gradient_(x, y)
            self.theta -= self.alpha * new_gradient
        return self.theta 

# m * n
def check_valid(array):
    if not isinstance(array, np.ndarray) or array.size == 0:
        return None
    if array.ndim == 2:
        return array
    return None

# m * 1
def check_valid_1(array):
    if not isinstance(array, np.ndarray) or array.size == 0:
        return None
    if array.ndim == 1:
        return array.reshape(array.size, 1)
    elif array.ndim == 2 or array.shape[1] == 1:
        return array
    return None

# (n + 1) * 1
def check_valid_theta(array, col_x):
    if not isinstance(array, np.ndarray) or array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(array.size, 1)
    elif array.ndim != 2 or array.shape[1] != 1:
        return None
    if array.shape[0] != col_x + 1:
        return None
    return array

if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)
    # Example 0:
    print(mylr.predict_(X))
    # # Output:
    # array([[0.99930437],
    #        [1.],
    #        [1.]])

    # Example 1:
    # print(np.mean(mylr.loss_elem_(Y, y_hat)))
    y_hat = mylr.predict_(X)
    print(mylr.loss_(Y, y_hat))
    # print(mylr.loss_(X, Y))
    # # Output:
    # 11.513157421577004
    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    # # # Output:
    # # array([[2.11826435]
    # #        [0.10154334]
    # #        [6.43942899]
    # #        [-5.10817488]
    # #        [0.6212541]])
    # # Example 3:
    print(mylr.predict_(X))
    # # # Output:
    # # array([[0.57606717]
    # #        [0.68599807]
    # #        [0.06562156]])
    # # Example 4:
    y_hat = mylr.predict_(X)
    print(mylr.loss_(Y, y_hat))
    # # # Output:
    # # 1.4779126923052268

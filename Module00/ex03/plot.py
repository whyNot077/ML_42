import numpy as np
import matplotlib.pyplot as plt

def add_intercept(x):
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    one = np.ones((x.shape[0], 1))
    return np.concatenate((one, x), axis=1)

def predict_(x, theta):
    X = add_intercept(x)
    y_hat = np.dot(X, theta)
    return y_hat

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array. 
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
            Nothing.
        Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or x.size == 0 or x.ndim != 1 or \
        not isinstance(y, np.ndarray) or y.size == 0 or y.ndim != 1 or \
        not isinstance(theta, np.ndarray) or theta.size == 0 or theta.shape != (2, 1):
        return None
    
    if x.size != y.size:
        return None
    plt.scatter(x, y)

    y_hat = predict_(x, theta)
    if y_hat is None:
        return None
    plt.plot(x, y_hat, color='orange')
    
    plt.show()

if __name__ == "__main__":
    x = np.arange(1,6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    # Example 1:
    theta1 = np.array([[4.5],[-0.2]])
    plot(x, y, theta1)

    # Example 2:
    theta2 = np.array([[-1.5],[2]])
    plot(x, y, theta2)

    # Example 3:
    theta3 = np.array([[3],[0.3]])
    plot(x, y, theta3)
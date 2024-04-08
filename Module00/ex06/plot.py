import numpy as np
import matplotlib.pyplot as plt

def loss_dot(y, y_hat):
    if not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
        y.ndim != y_hat.ndim:
        return None
    
    cost = np.sum(np.dot((y_hat - y), (y_hat - y))) / y.size
    return cost

def add_intercept(x):
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    one = np.ones((x.shape[0], 1))
    return np.concatenate((one, x), axis=1)

def predict_(x, theta):
    X = add_intercept(x)
    y_hat = np.dot(X, theta)
    return y_hat

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
        Returns:
            Nothing.
        Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0 or \
        not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(theta, np.ndarray) or theta.size == 0 or \
        y.size != x.size or theta.shape != (2,):
        return None

    plt.scatter(x, y)
    y_hat = predict_(x, theta)
    if y_hat is None:
        return None
    plt.plot(x, y_hat, color='orange')

    lose = loss_dot(y, y_hat)
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], linestyle='--', color='red')

    plt.title("Cost: {:.6f}".format(lose))
    plt.show()

if __name__ == "__main__":
    x = np.arange(1,6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    # Example 1:
    theta1= np.array([18,-1]) 
    plot_with_loss(x, y, theta1)

    # Example 2:
    theta2 = np.array([14, 0]) 
    plot_with_loss(x, y, theta2)

    # Example 3:
    theta3 = np.array([12, 0.8]) 
    plot_with_loss(x, y, theta3)
import numpy as np


def reg_linear_grad(y, x, theta, lambda_):
    """ Computes the regularized linear gradient of three non-empty numpy.ndarray,
	with two for-loop. The three arrays must have compatible shapes.
	Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
	Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
	Raises:
	    This function should not raise any Exception.
	"""
    check_valid_1(y)
    check_valid(x)
    check_valid_theta(theta, x.shape[1])
    if x is None or y is None or theta is None or \
    x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None

    m = x.shape[0]
    n = x.shape[1]
    gradient = np.zeros((n + 1, 1))
    X = np.hstack((np.ones((m, 1)), x))

    for i in range(m):
        xi = X[i].reshape(n + 1, 1).astype(float)
        yi = y[i]
        hypothesis = np.dot(theta.T, xi)[0, 0]
        gradient += (hypothesis - yi) * xi

    gradient /= m
    gradient[1:] += (lambda_ / m) * theta[1:]

    return gradient

def vec_reg_linear_grad(y, x, theta, lambda_):
    """
	Computes the regularized linear gradient of three non-empty numpy.ndarray,
	without any for-loop. The three arrays must have compatible shapes.
	Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
	Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
	Raises:
	    This function should not raise any Exception.
	"""
    check_valid_1(y)
    check_valid(x)
    check_valid_theta(theta, x.shape[1])
    if x is None or y is None or theta is None or \
    x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None
    
    m = x.shape[0]
    X = np.hstack((np.ones((m, 1)), x))
    h_theta = np.dot(X, theta)

    gradient = (1 / m) * np.dot(X.T, h_theta - y)
    gradient[1:] += (lambda_ / m) * theta[1:]

    return gradient


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
    elif array.ndim != 2 or array.shape[1] != 1:
        return None
    return array
    
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
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    # Example 1.1:
    print(reg_linear_grad(y, x, theta, 1))
    # # Output:
    # array([[-60.99],
    #        [-195.64714286],
    #        [863.46571429],
    #        [-644.52142857]])
    # Example 1.2:
    print(vec_reg_linear_grad(y, x, theta, 1))
    # # Output:
    # array([[-60.99],
    #        [-195.64714286],
    #        [863.46571429],
    #        [-644.52142857]])
    # Example 2.1:
    print(reg_linear_grad(y, x, theta, 0.5))
    # # Output:
    # array([[-60.99],
    #        [-195.86142857],
    #        [862.71571429],
    #        [-644.09285714]])
    # Example 2.2:
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # # Output:
    # array([[-60.99],
    #        [-195.86142857],
    #        [862.71571429],
    #        [-644.09285714]])
    # Example 3.1:
    print(reg_linear_grad(y, x, theta, 0.0))
    # # Output:
    # array([[-60.99],
    #        [-196.07571429],
    #        [861.96571429],
    #        [-643.66428571]])
    # Example 3.2:
    print(vec_reg_linear_grad(y, x, theta, 0.0))
    # # Output:
    # array([[-60.99],
    #        [-196.07571429],
    #        [861.96571429],
    #        [-643.66428571]])

import numpy as np


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model
    from two non-empty numpy.array, without any for loop.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    check_valid_1(y)
    check_valid_1(y_hat)
    check_valid_1(theta)
    if y is None or y_hat is None or theta is None or \
        not isinstance(lambda_, float) or \
        y.shape != y_hat.shape:
        return None

    
    loss = np.dot((y_hat - y).T, (y_hat - y))
    regularization_term = lambda_ * np.dot(theta[1:].T, theta[1:])
    return float(1 / (2 * len(y)) * (loss + regularization_term))

# m * 1
def check_valid_1(array):
    if not isinstance(array, np.ndarray) or array.size == 0:
        return None
    if array.ndim == 1:
        return array.reshape(array.size, 1)
    elif array.ndim != 2 or array.shape[1] != 1:
        return None
    return array

if __name__ == "__main__":
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    # Example :
    print(reg_loss_(y, y_hat, theta, .5))
    # # Output:
    # 0.8503571428571429
    # Example :
    print(reg_loss_(y, y_hat, theta, .05))
    # # Output:
    # 0.5511071428571429
    # Example :
    print(reg_loss_(y, y_hat, theta, .9))
    # # Output:
    # 1.116357142857143

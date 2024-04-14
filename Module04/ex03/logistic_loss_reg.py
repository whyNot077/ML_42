import numpy as np


def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model
    from two non-empty numpy.ndarray, without any for loop.
    Args:
        y: has to be a numpy.ndarray, a vector of shape (m, 1).
        y_hat: has to be a numpy.ndarray, a vector of shape (m, 1).
        theta: has to be a numpy.ndarray, a vector of shape (n, 1).
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
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

    m = len(y)
    y = np.array(y).reshape(-1, 1)
    y_hat = np.array(y_hat).reshape(-1, 1)
    theta = np.array(theta).reshape(-1, 1)

    term1 = np.dot(y.T, np.log(y_hat))
    term2 = np.dot((1 - y).T, np.log(1 - y_hat))
    regularization = (lambda_ / (2 * m)) * np.dot(theta[1:].T, theta[1:])
    loss = -((term1 + term2) / m) + regularization

    return loss.item()

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
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    lambda_ = 0.5

    # Example :
    # Output: 0.43377043716475955
    print(reg_log_loss_(y, y_hat, theta, lambda_))

    lambda_ = 0.05
    # Example :
    # Output: 0.13452043716475953
    print(reg_log_loss_(y, y_hat, theta, lambda_))

    lambda_ = 0.9
    # Example :
    # Output: 0.6997704371647596
    print(reg_log_loss_(y, y_hat, theta, lambda_))

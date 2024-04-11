import numpy as np
    
def vec_log_loss_(y, y_hat, eps=1e-15): 
    """
        Compute the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
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

# m * 1
def check_valid_1(array):
    if not isinstance(array, np.ndarray) or array.size == 0:
        return None
    if array.ndim == 1:
        return array.reshape(array.size, 1)
    elif array.ndim != 2 or array.shape[1] != 1:
        return None
    return array

def logistic_predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.size == 0 or \
    not isinstance(theta, np.ndarray) or theta.size == 0 or \
    not x.ndim == 2:
        return None
    
    if theta.ndim == 1 and theta.size == x.shape[1] + 1:
        theta = theta.reshape(x.shape[1] + 1, 1)
    elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
        return None

    modified_x = np.hstack((np.ones((x.shape[0], 1)), x))
    return np.array(1 / (1 + np.exp(-modified_x.dot(theta))))


if __name__ == "__main__":
    # Example 1:
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(vec_log_loss_(y1, y_hat1))
    # # Output:
    # 0.018149927917808714

    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(vec_log_loss_(y2, y_hat2))

    # # Output:
    # 2.4825011602472347
    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(vec_log_loss_(y3, y_hat3))
    # # Output:
    # 2.993853310859968

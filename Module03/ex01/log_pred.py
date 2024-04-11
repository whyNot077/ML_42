import numpy as np
    
def logistic_predict_(x, theta):
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
    theta = check_valid_theta(theta, x.shape[1])
    if theta is None:
        return None
    modified_x = np.hstack((np.ones((x.shape[0], 1)), x))
    return np.array(1 / (1 + np.exp(-modified_x.dot(theta))))


# m * n
def check_valid(array):
    if not isinstance(array, np.ndarray) or array.size == 0:
        return None
    if array.ndim == 2:
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


def print_array(x):
    print("array(", end="")
    for idx, item in enumerate(x):
        if idx < len(x) - 1:
            print(f"{item}, ", end="")
        else:
            print(f"{item}", end="")
    print(")")

if __name__ == "__main__":
    # Example 1
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])

    print_array(logistic_predict_(x, theta))
    # # Output:
    # array([[0.98201379]])
    # Example 1

    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print_array(logistic_predict_(x2, theta2))
    # # Output:
    # array([[0.98201379],
    #        [0.99624161],
    #        [0.97340301],
    #        [0.99875204],
    #        [0.90720705]])
    # Example 3
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print_array(logistic_predict_(x3, theta3))
    # # Output:
    # array([[0.03916572],
    #        [0.00045262],
    #        [0.2890505]])

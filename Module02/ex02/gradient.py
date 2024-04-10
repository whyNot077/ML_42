import numpy as np

def print_array(x):
    print("array(", end="")
    for idx, item in enumerate(x):
        if idx < len(x) - 1:
            print(f"{item}, ", end="")
        else:
            print(f"{item}", end="")
    print(")")

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop. The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
            containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0 or x.ndim != 2 or\
        not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(theta, np.ndarray) or theta.size == 0 or\
            x.shape[0] != y.shape[0]:
        return None
    if x.shape[1] + 1 != theta.shape[0]:
        theta = np.insert(theta, 0, 0, axis=0)
    
    m = x.shape[0]
    modified_x = np.hstack([np.ones((m, 1)), x])
    transpose_x = modified_x.T    
    expected_y = np.dot(modified_x, theta)
    disparity = expected_y - y
    j = np.dot(transpose_x, disparity) / m
    print_array(j)
    return j
    

if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    theta1 = np.array([3,0.5,-6]).reshape((-1, 1))
    # Example :
    gradient(x, y, theta1)
    # Output:
    # array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])
    # Example :
    theta2 = np.array([0,0,0]).reshape((-1, 1))
    gradient(x, y, theta2)
    # Output:
    # array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])
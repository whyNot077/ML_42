def add_polynomial_features(x, power):
    """ Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            power: has to be an int, the power up to which the columns of matrix x are going to be raised.
        Returns:
            The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
            None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
    """
    check_valid(x)
    if x is None or not isinstance(power, int):
        return None
    
    res = x
    for pow in range(2, power + 1):
        res = np.hstack((res, np.power(x, pow)))

    return res
        


# m * n
def check_valid(array):
    if not isinstance(array, np.ndarray) or array.size == 0:
        return None
    if array.ndim == 2:
        return array
    return None

if __name__ == "__main__":
    import numpy as np
    x = np.arange(1, 11).reshape(5, 2)
    # print(x)
    # Example 1:
    print(add_polynomial_features(x, 3))
    # # Output:
    # array([[1, 2, 1, 4, 1, 8],
    #        [3, 4, 9, 16, 27, 64],
    #        [5, 6, 25, 36, 125, 216],
    #        [7, 8, 49, 64, 343, 512],
    #        [9, 10, 81, 100, 729, 1000]])
    # Example 2:
    print(add_polynomial_features(x, 4))
    # # Output:
    # array([[1, 2, 1, 4, 1, 8, 1, 16],
    #        [3, 4, 9, 16, 27, 64, 81, 256],
    #        [5, 6, 25, 36, 125, 216, 625, 1296],
    #        [7, 8, 49, 64, 343, 512, 2401, 4096],
    #        [9, 10, 81, 100, 729, 1000, 6561, 10000]])

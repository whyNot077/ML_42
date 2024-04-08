import numpy as np

def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) and x.size == 0:
        return None
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    one = np.ones((x.shape[0], 1))
    # print(f"array({np.hstack((intercept, x))})")

    res = np.concatenate((one, x), axis=1)
    print(f"array({res})")
    return res

if __name__ == "__main__":
    # Example 1:
    x = np.arange(1,6) 
    # print(f"\nI'm x. I have {x.ndim} dimensions \n({x})")
    add_intercept(x)
    # Output: array([[1., 1.],
        #    [1., 2.],
        #    [1., 3.],
        #    [1., 4.],
        #    [1., 5.]])
    # Example 2:
    y = np.arange(1,10).reshape((3,3)) 
    # print(f"\nI'm y. I have {y.ndim} dimensions \n({y})")
    add_intercept(y)
    # Output:
    # array([[1., 1., 2., 3.],
    #        [1., 4., 5., 6.],
    #        [1., 7., 8., 9.]])
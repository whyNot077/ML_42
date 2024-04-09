import numpy as np
import math

def mean(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None

    return np.sum(x) / x.size

def std(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    
    mean_x = mean(x)
    var = np.sum((x - mean_x) ** 2) / (x.size - 1)
    std = np.sqrt(var)

    return(std)

def zscore(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization. Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn’t raise any Exception.
    """

    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    
    mean_x = mean(x)
    std_x = std(x)
    res = (x - mean_x) / std_x
    print(res)
    return res

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    zscore(X)
    # Output:
    # array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559,
                    # 0.17240647, -1.89647119])
    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    zscore(Y)
    # Output:
    # array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659,
                    # 0.28795027, -1.72770165])
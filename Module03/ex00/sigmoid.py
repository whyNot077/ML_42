import numpy as np
import math

def print_array(x):
    print("array(", end="")
    for idx, item in enumerate(x):
        if idx < len(x) - 1:
            print(f"{item}, ", end="")
        else:
            print(f"{item}", end="")
    print(")")

def sigmoid_(x): 
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    
    return np.array(1 / (1 + np.exp(-x)))

if __name__ == "__main__":
    # Example 1:
    x = np.array([[-4]])
    print_array(sigmoid_(x))
    # # Output:
    # array([[0.01798620996209156]])
    # Example 2:
    x = np.array([[2]])
    print_array(sigmoid_(x))
    # # Output:
    # array([[0.8807970779778823]])
    # Example 3:
    x = np.array([[-4], [2], [0]])
    print_array(sigmoid_(x))
    # Output:
    # array([[0.01798620996209156], [0.8807970779778823], [0.5]])

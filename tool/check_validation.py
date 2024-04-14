import numpy as np

def print_array(x):
    print("array(", end="")
    for idx, item in enumerate(x):
        if idx < len(x) - 1:
            print(f"{item}, ", end="")
        else:
            print(f"{item}", end="")
    print(")")

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

def get_biased_x(x):
    return np.hstack((np.ones((x.shape[0], 1)), x))

if __name__ == "__main__":
    y=np.array([[0], [0], [0]])
    y_hat=np.array([[1], [0], [1]])

    v = [y, y_hat]
    for i in range(len(v)):
        v[i] = check_valid_1(v[i])
        if v[i] is None:
            print("None")
    y, y_hat = v
    
    print(y)
    print(y_hat)
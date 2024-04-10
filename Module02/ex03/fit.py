import numpy as np

def print_array(x):
    print("array(", end="")
    for idx, item in enumerate(x):
        if idx < len(x) - 1:
            print(f"{item}, ", end="")
        else:
            print(f"{item}", end="")
    print(")")

def predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.size == 0 or x.ndim != 2 or \
        not isinstance(theta, np.ndarray) or theta.size == 0 or \
            x.shape[1] + 1 != theta.shape[0]:
        return None
    
    x_modified = np.hstack((np.ones((x.shape[0], 1)), x))
    y_hat = np.dot(x_modified, theta)
    print_array(y_hat)
    return y_hat

def gradient(x, y, theta):   
    m = x.shape[0]
    modified_x = np.hstack([np.ones((m, 1)), x])
    transpose_x = modified_x.T    
    expected_y = np.dot(modified_x, theta)
    disparity = expected_y - y
    j = np.dot(transpose_x, disparity) / m
    return j
    
def fit_(x, y, theta, alpha, max_iter): 
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
                    (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
                    (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
                    (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0 or x.ndim != 2 or\
        not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(theta, np.ndarray) or theta.size == 0 or\
        not isinstance(alpha, float) or \
        not isinstance(max_iter, int) or \
        x.shape[0] != y.shape[0]:
        return None
    if x.shape[1] + 1 != theta.shape[0]:
        theta = np.insert(theta, 0, 0, axis=0)

    new_theta = theta.astype(np.float64).copy()
    for _ in range(max_iter):
        new_gradient = gradient(x, y, new_theta)
        new_theta -= alpha * new_gradient

    print_array(new_theta)
    return new_theta

if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]]) 
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000) 
    # Output:
    # array([[41.99..],[0.97..], [0.77..], [-1.20..]])
    # Example 1:
    predict_(x, theta2)
    # Output:
    # array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
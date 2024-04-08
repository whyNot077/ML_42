import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
    """
    Description:
            Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
        y.size != y_hat.size:
        return None
    
    mse = np.sum((y_hat - y) ** 2) / y.size
    print(mse)
    return mse

def rmse_(y, y_hat): 
    """
    Description:
            Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
        y.size != y_hat.size:
        return None
    
    rmse = sqrt(np.sum((y_hat - y) ** 2) / y.size)
    print(rmse)
    return rmse
    
def mae_(y, y_hat): 
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.  
    """
    if not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
        y.size != y_hat.size:
        return None
    
    mae = np.sum(abs(y_hat - y)) / y.size
    print(mae)
    return mae

def r2score_(y, y_hat): 
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.  
    """
    if not isinstance(y, np.ndarray) or y.size == 0 or \
        not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or \
        y.size != y_hat.size:
        return None
    
    deno = np.sum((y_hat - y) ** 2)
    nume = np.sum((y - np.mean(y)) ** 2)
    rto_score = 1 - (deno / nume)
    print(rto_score)
    return rto_score

    
if __name__ == "__main__":

    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    # Mean squared error
    ## your implementation 
    mse_(x,y)
    ## Output: 4.285714285714286
    ## sklearn implementation 
    mean_squared_error(x,y) 
    # ## Output: 4.285714285714286
    # # Root mean squared error 
    # ## your implementation 
    rmse_(x,y)
    # ## Output: 2.0701966780270626
    # ## sklearn implementation not available: take the square root of MSE
    sqrt(mean_squared_error(x,y))
    # ## Output:
    # # 2.0701966780270626
    # # Mean absolute error 
    # ## your implementation 
    mae_(x,y)
    # # Output: 1.7142857142857142
    # ## sklearn implementation
    mean_absolute_error(x,y)
    # # Output:
    # # 1.7142857142857142
    # # R2-score
    # ## your implementation 
    r2score_(x,y)
    # ## Output: 0.9681721733858745
    ## (It can differ from the output due to precision issues in floating-point arithmetic.)
    # # sklearn implementation 
    r2_score(x,y)
    # ## Output: 0.9681721733858745
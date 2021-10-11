import numpy as np

def normalizeRow(x):
    # return np.divide(x, np.reshape(np.sqrt(np.sum(x**2, axis=1)), (x.shape[0],1)))
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    
    # Divide x by its norm.
    return x/x_norm


x = np.array([[0,3,4, 2],[1,6,4, 3],[31556, 45646, 33231, 85184]])
print(normalizeRow(x))
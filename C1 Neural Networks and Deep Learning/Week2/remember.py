import h5py
import numpy as np

def getData():
    with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('datasets/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    

def normalization(x):
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    return x/x_norm

def initialization(n, m):
    W = np.random.randn(n, 1) * 0.001
    b = 0
    parameters = {"W" : W, "b" : b}
    return parameters

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def forward_propagation(X, Y, m, parameters):
    W = parameters["W"]
    b = parameters["b"]
    z = np.dot(W.T, X) + b
    activation = sigmoid(z)
    cost = (-1/m) * np.sum(Y * np.log(activation) + (1 - Y) * np.log(1 - activation))
    return {"activation" : activation, "cost" : cost}

def backward_propagation(X, Y, m, parameters, activation, alpha):
    dz = activation - Y
    dw = (1/m) * np.dot(X, dz.T)
    db = (1/m) * np.sum(dz)

    update_param = {"W" : parameters["W"] - (alpha * dw), 
                    "b" : parameters["b"] - (alpha * db)}

    return update_param



def model(X, Y, alpha = 0.0001, iter = 300):
    n = X.shape[0]
    m = X.shape[1]

    # X = normalization(X)
    parameters = initialization(n, m)

    for i in range(iter):
        res = forward_propagation(X, Y, m, parameters)
        parameters = backward_propagation(X, Y, m, parameters, res["activation"], alpha)

        if i % 100 == 0 :
            print(res["cost"])


def test():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = getData()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    model(train_set_x, train_set_y, alpha= 0.005, iter = 2000)

test()
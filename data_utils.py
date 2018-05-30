import numpy as np
from sklearn.preprocessing import PolynomialFeatures, normalize

def loadData():
    X = np.loadtxt("Xdata.txt")
    y = np.reshape(np.loadtxt("ydata.txt"), (X.shape[0], 1))
    
    D = np.concatenate((X, y), axis=1)
    
    np.random.shuffle(D)
        
    X = PolynomialFeatures(2, interaction_only=False, include_bias=False).fit_transform(D[...,:-1], )
    #X = normalize(X, axis=0)
    y = D[...,-1:]
    
    print("X shape = ", X.shape)
    print("y shape = ", y.shape)
    
    return X, y

def partitionSamples(X, y):
    numSamples = X.shape[0]
    trainingSamples = int(0.6 * numSamples)
    devSamples = int(0.2 * numSamples)
    testSamples = numSamples - trainingSamples - devSamples

    print("Total = {:d}, Training samples = {:d}, dev samples = {:d}, test samples = {:d}\n".format(numSamples, trainingSamples, devSamples, testSamples))

    X_train = X[0:trainingSamples, ...]
    y_train = y[0:trainingSamples, ...]

    X_dev = X[trainingSamples:trainingSamples+devSamples, ...]
    y_dev = y[trainingSamples:trainingSamples+devSamples, ...]

    X_test = X[-testSamples:, ...]
    y_test = y[-testSamples:, ...]

    return {"X_train":X_train, "y_train":y_train, "X_dev":X_dev, "y_dev":y_dev, "X_test":X_test, "y_test":y_test}

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

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.model_selection import train_test_split

def loadData():
    X = np.loadtxt("Xdata.txt")
    y = np.reshape(np.loadtxt("ydata.txt"), (X.shape[0], 1))
    
    X = PolynomialFeatures(2, interaction_only=False, include_bias=False).fit_transform(X)
    
    print("X shape = ", X.shape)
    print("y shape = ", y.shape)
    
    return X, y

def partitionSamples(X, y):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.4, shuffle=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, shuffle=False)

    print("Total = {:d}, Training samples = {:d}, dev samples = {:d}, test samples = {:d}\n".format(X.shape[0], X_train.shape[0], X_dev.shape[0], X_test.shape[0]))

    return {"X_train":X_train, "y_train":y_train, "X_dev":X_dev, "y_dev":y_dev, "X_test":X_test, "y_test":y_test}

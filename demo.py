from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve

from plot_utils import *
from data_utils import *
from model import *
   
# read 2d classification dataset
X, y = loadData()

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

num_classes = 2
model = buildModel((X.shape[1],), num_classes, [5, 5])

opt = Adam(decay=0.0005, amsgrad=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy', 'binary_accuracy'])

history = model.fit(X_train, to_categorical(y_train, num_classes=num_classes), validation_data=(X_dev, to_categorical(y_dev, num_classes=num_classes)), epochs=10, batch_size=100)

plotTraining(history)

#plotKernelWeights(model, "input_layer0")
 
print("test results = ", model.evaluate(X_test, to_categorical(y_test, num_classes=num_classes), batch_size=100), '\n')

y_pred_vec = model.predict(X_test)
print("prediction vectors", y_pred_vec[:5,...])

y_pred = np.argmax(y_pred_vec, axis=1)
#print("prediction", y_pred[:5,...])

print("F1 score = ", f1_score(y_test, y_pred), "MCC score =", matthews_corrcoef(y_test, y_pred))

#print(classification_report(y_test, y_pred))

#print("Confusion matrix =\n", confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_pred_vec[:,1])
plotROC(fpr, tpr)

#print("prediction on ", X_test[0,...], " is ", y_pred, "; ground truth is ", to_categorical(y_test[0:1,...], num_classes=num_classes))

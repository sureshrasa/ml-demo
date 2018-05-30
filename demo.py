from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve

from plot_utils import *
from data_utils import *
from model import *
   
# read 2d classification dataset
X, y = loadData()

dataSets = partitionSamples(X, y)

X_train = dataSets["X_train"]
y_train = dataSets["y_train"]
X_dev = dataSets["X_dev"]
y_dev = dataSets["y_dev"]
X_test = dataSets["X_test"]
y_test = dataSets["y_test"]

num_classes = 2
model = buildModel((X.shape[1],), num_classes, [5, 5])

opt = Adam(decay=0.0005, amsgrad=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy', 'binary_accuracy'])

history = model.fit(X_train, to_categorical(y_train, num_classes=num_classes), validation_data=(X_dev, to_categorical(y_dev, num_classes=num_classes)), epochs=15, batch_size=100)

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

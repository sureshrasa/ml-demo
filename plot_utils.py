import numpy as np
from matplotlib import pyplot as plt
from keras.utils import plot_model

def plotTraining(history):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_cat', 'dev_cat', 'train_bin', 'dev_bin'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='lower right')
    plt.show()
    
def plotWeights(title, weights):
    legend = []
    for i, w in enumerate(weights):
        plt.plot(w)
        legend.append("W"+str(i))
        print("W{:d} shape = ".format(i), w.shape)
        
    plt.title(title)
    plt.ylabel('value')
    plt.xlabel('unit')
    plt.legend(legend, loc='upper right')
    plt.show()

def plotKernelWeights(model, layerName):
    W = model.get_layer(name=layerName).get_weights();
    plotWeights("model weights", W[0])
    plotWeights("model bias", W[1])

def plotROC(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

def plotModel(model):
    plot_model(model, to_file='model.png')
    print(model.get_weights()[0].shape)

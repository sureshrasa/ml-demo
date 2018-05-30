import numpy as np

from keras.models import Model
from keras.layers import Dense, Activation, BatchNormalization
from keras.engine.topology import Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, RMSprop
import keras.backend as K

def buildModel(inputShape, num_classes, layers):
    print("Building model for input shape = ", inputShape, "with layers =", layers)
    X0 = Input(shape=inputShape)
    X = X0
    X = BatchNormalization(input_shape=inputShape)(X)
  
    for i in range(len(layers)):
        X = Dense(layers[i], activation='relu', name="input_layer"+str(i))(X)
        #X = BatchNormalization(input_shape=inputShape)(X)
        
    X = Dense(num_classes, name="output_layer")(X)
    X = Activation('softmax')(X)
    
    model = Model(inputs=[X0], outputs=[X])
    
    #print(model.to_json())
    return model

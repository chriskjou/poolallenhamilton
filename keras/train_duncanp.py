from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.models import model_from_json, model_from_yaml
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../')
from readercleaner import get_dataduncanp

# rather, for pool
output_dim = 3
model_dir = 'duncanp_model.json'
weights_dir = 'duncanp_wts.h5'
train_data = get_dataduncanp(0,290)
X_train = train_data[0]
Y_train = train_data[1]
test_data = get_dataduncanp(290,340)
X_test = test_data[0]
Y_test = test_data[1]

Y_train += 1
Y_test += 1
# is it ok if i don't convert class vectors to binary class matrices?
Y_train = np_utils.to_categorical(Y_train, 3)
Y_test = np_utils.to_categorical(Y_test, 3)

print("X size", X_train.shape)
print("Y size", Y_train.shape)

# build the model

# X_train = np.expand_dims(X_train, axis=2)
# X_test = np.expand_dims(X_test, axis=2)

model = Sequential()
# model.add(Conv1D(1, 3, strides=3, input_shape = (45,1), padding = 'same', activation = 'sigmoid'))
# model.add(Flatten())
model.add(Dense(15,input_dim = 45, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(output_dim, activation='softmax'))
model.summary()
batch_size = 32
nb_epoch = 20

# compile the model

sgd = optimizers.SGD(lr=0.01) # added a higher learning rate
model.compile(optimizer=sgd, loss='mean_absolute_error', metrics=['accuracy']) # changed optim, error
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# save model and weights

json_string = model.to_json() # as json
open(model_dir, 'w').write(json_string)
# yaml_string = model.to_yaml() #as yaml
# open('mnist_Logistic_model.yaml', 'w').write(yaml_string)

# save the weights in h5 format
model.save_weights(weights_dir)

# uncomment the code below (and modify accordingly) to read a saved model and weights
model = model_from_json(open(model_dir).read())# if json
# model = model_from_yaml(open('mnist_Logistic_model.yaml').read())# if yaml
model.load_weights(weights_dir)



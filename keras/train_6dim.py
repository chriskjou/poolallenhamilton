from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json, model_from_yaml
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from readercleaner import get_data2

# rather, for pool
input_dim = 6
output_dim = 1
model_dir = 'trunc_6dim_model.json'
weights_dir = 'trunc_6dim_wts.h5'
train_data = get_data2(0,290)
X_train = train_data[['easystripe','easysolid','medstripe','medsolid','hardstripe','hardsolid']].as_matrix()
Y_train = train_data[['winner']].as_matrix()
test_data = get_data2(290,340)
X_test = test_data[['easystripe','easysolid','medstripe','medsolid','hardstripe','hardsolid']].as_matrix()
Y_test = test_data[['winner']].as_matrix()

# is it ok if i don't convert class vectors to binary class matrices?
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

print("X size", X_train.shape)
print("Y size", Y_train.shape)

# build the model

model = Sequential()
model.add(Dense(12, input_dim=input_dim, activation='sigmoid')) # changed from softmax
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(output_dim, activation='sigmoid'))
model.summary()
batch_size = 64
nb_epoch = 40

# compile the model

sgd = optimizers.SGD(lr=0.01) # added a higher learning rate
model.compile(optimizer=sgd, loss='mean_absolute_error', metrics=['accuracy']) # changed optim, error
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# save losses
loss_history = history.history["loss"]
np_loss_history = np.array(loss_history)
print(np_loss_history.shape)
np.save('6dim_history',np_loss_history)

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



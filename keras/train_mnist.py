from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json, model_from_yaml
import matplotlib.pyplot as plt

# the data, shuffled and split between train and test sets

(X_train, y_train), (X_test, y_test) = mnist.load_data()

input_dim = 784 #28*28
output_dim = nb_classes = 10
X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print("X size", X_train.shape)
print("Y size", Y_train.shape)

# build the model

model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
model.summary()
batch_size = 128
nb_epoch = 20

# compile the model

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# save model and weights

json_string = model.to_json() # as json
open('mnist_Logistic_model.json', 'w').write(json_string)
# yaml_string = model.to_yaml() #as yaml
# open('mnist_Logistic_model.yaml', 'w').write(yaml_string)

# save the weights in h5 format
model.save_weights('mnist_Logistic_wts.h5')

# uncomment the code below (and modify accordingly) to read a saved model and weights
model = model_from_json(open('mnist_Logistic_model.json').read())# if json
# model = model_from_yaml(open('mnist_Logistic_model.yaml').read())# if yaml
model.load_weights('mnist_Logistic_wts.h5')




















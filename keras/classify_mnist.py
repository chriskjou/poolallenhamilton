from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json, model_from_yaml
import matplotlib.pyplot as plt

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




model = model_from_json(open('mnist_Logistic_model.json').read())# if json
# model = model_from_yaml(open('mnist_Logistic_model.yaml').read())# if yaml
model.load_weights('mnist_Logistic_wts.h5')




test = X_test[0].reshape(-1,784)
print("result is",model.predict(test))
plt.imshow(test.reshape(28,28))
plt.show()

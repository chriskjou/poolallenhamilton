from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json, model_from_yaml
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, '../')
from readercleaner import get_data

# rather, for pool
output_dim = 1
model_dir = 'simple_model.json'
weights_dir = 'simple_wts.h5'
test_data = get_data(200,250)
X_test = test_data[['x','y','cuex','cuey']].as_matrix()
Y_test = test_data[['diff']].as_matrix()

# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

print("X size", X_test.shape)
print("Y size", Y_test.shape)



model = model_from_json(open(model_dir).read())# if json
# model = model_from_yaml(open('mnist_Logistic_model.yaml').read())# if yaml
model.load_weights(weights_dir)

print("result is",model.predict(X_test))

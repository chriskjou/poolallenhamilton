from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json, model_from_yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from readercleaner import get_data1

# rather, for pool
input_dim = 2
hidden_dim = 5
output_dim = 1
model_dir = '2dim_model.json'
weights_dir = '2dim_wts.h5'
test_data = get_data1(20,21)
X_test = test_data[['numstripe','numsolid']].as_matrix()
Y_test = test_data[['winner']].as_matrix()

# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

print("X size", X_test.shape)
print("Y size", Y_test.shape)

model = model_from_json(open(model_dir).read())# if json
# model = model_from_yaml(open('mnist_Logistic_model.yaml').read())# if yaml
model.load_weights(weights_dir)

print("winner is", Y_test[0])
print("result is",model.predict(X_test))

###############

heatmap = np.zeros((8,8))
for numsolid in range(8):
    for numstripe in range(8):
        x = np.array([numstripe,numsolid])
        # a previous iteration of the keras model confusingly broke the "solids first" paradigm
        heatmap[numsolid,numstripe] = model.predict(x[np.newaxis])[0][0]

plt.imshow(heatmap, origin='lower')
plt.colorbar()
plt.title('Network Predictions (1 indicates stripe victory)')
plt.xlabel('# solids on table')
plt.ylabel('# stripes on table')
plt.show()

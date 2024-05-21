import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime
from keras import utils

import pandas as pd
from sklearn.model_selection import train_test_split

# load data on laptop
#mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Table_out.csv")

# load data on computer
mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Code Python\Table_out.csv")

# separate data and target
data, target = mimic.drop(columns=["SUBJECT_ID","EXPIRE_FLAG", "'58177000111'"]), mimic["EXPIRE_FLAG"]

data = data.to_numpy()
target = target.to_numpy()

#print(data.shape)

# split data_train and the data_test
x_train, x_test, y_train, y_test = train_test_split(
    data, target, random_state=42, test_size=0.33
)


# reshape input data
x_train = x_train.reshape(x_train.shape[0], 1, 88)


# same for test data
x_test = x_test.reshape(x_test.shape[0], 1, 88)

# Network
net = Network()
net.add(FCLayer(88, 64))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(64, 32))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(32, 8))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(8, 1))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))


net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test)
#print("\n")
#print("predicted values : ")
#print(out, end="\n")
#print("true values : ")
#print(y_test)

good = 0
for i in range(len(x_test)):
    if out[i] == y_test[i]:
        good += 1
print(good)
print("accuracy:", good/len(x_test))


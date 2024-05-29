import keras
import tensorflow as tf
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

# load data on laptop
# mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Table_out.csv")

# load data on computer
mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Code Python\Table_out.csv")

# separate data and target
data, target = mimic.drop(columns=["SUBJECT_ID", "EXPIRE_FLAG", "'58177000111'"]), mimic["EXPIRE_FLAG"]

data = data.to_numpy()
target = target.to_numpy()

# split data_train and the data_test
x_train, x_test, y_train, y_test = train_test_split(
    data, target, random_state=42, test_size=0.33
)

# reshape input data
x_train = x_train.reshape(x_train.shape[0], 1, 88)
y_train = y_train.reshape(y_train.shape[0], 1)
print(y_train)

# same for test data
x_test = x_test.reshape(x_test.shape[0], 1, 88)
y_test = y_test.reshape(y_test.shape[0], 1)

model = keras.Sequential()
model.add(keras.Input(shape=(1, 88)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.fit(x_train, y_train, epochs=100)
model_weights = model.get_weights()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")



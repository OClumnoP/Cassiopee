import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

# load data on laptop
mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Table_out.csv")

# load data on computer
#mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Code Python\Table_out.csv")

# separate data and target
data, target = mimic.drop(columns=["SUBJECT_ID", "EXPIRE_FLAG", "'58177000111'"]), mimic["EXPIRE_FLAG"]

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

model = keras.Sequential()
model.add(keras.Input(shape=(1, 88)))
model.add(layers.Dense(88, activation="relu"))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")


import keras
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt

# load data on laptop
# mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Table_out.csv")

# load data on computer
mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Code Python\Table_out.csv")

# separate data and target
data, target = mimic.drop(columns=["SUBJECT_ID", "EXPIRE_FLAG", "'58177000111'"]), mimic["EXPIRE_FLAG"]

data = data.to_numpy()
target = target.to_numpy()

# split data_train and the data_test
# x_train, x_test, y_train, y_test = train_test_split(
#     data, target, random_state=42, test_size=0.33)

# # reshape input data
# x_train = x_train.reshape(x_train.shape[0], 1, 88)
# y_train = y_train.reshape(y_train.shape[0], 1)
#
# # same for test data
# x_test = x_test.reshape(x_test.shape[0], 1, 88)
# y_test = y_test.reshape(y_test.shape[0], 1)

# reshape data
x = data.reshape(data.shape[0], 1, 88)
y = target.reshape(target.shape[0], 1)

model = keras.Sequential()
model.add(keras.Input(shape=(1, 88)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
#model.add(layers.Dense(8, activation="relu"))
#model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()]
)

history = model.fit(x, y, validation_split=0.33, epochs=100, shuffle=False)
model_weights = model.get_weights()

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('binary_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for recall
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# test_loss, test_acc, test_pre, test_rec = model.evaluate(x_test, y_test, verbose=2)
# print(f"Test binary_accuracy: {test_acc}")
# print(f"Test precision: {test_pre}")
# print(f"Test recall: {test_rec}")

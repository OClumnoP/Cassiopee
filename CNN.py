import keras
from keras import layers
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Table_out.csv")

#separate data and target
data, target = mimic.drop(columns=["EXPIRE_FLAG"]), mimic["EXPIRE_FLAG"]

#split data_train and the data_test
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42, test_size=0.33
)


layers_input = layers.Input(shape=(128,))


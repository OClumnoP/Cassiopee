import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# load database
mimic = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\Table_out.csv")

data, target = mimic.drop(columns=["EXPIRE_FLAG"]), mimic["EXPIRE_FLAG"]


data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42, test_size=0.25
)

model = LogisticRegression()
model.fit(data_train, target_train)
accuracy = model.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy:.3f}")
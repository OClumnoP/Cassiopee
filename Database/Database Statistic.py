import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Patient = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\mimic-iii-clinical-database\PATIENTS.csv")

Patient = Patient[['GENDER']]
Male = 0
Female = 0

for i in range(Patient.shape[0]):
    value = Patient.iat[i, 0]
    if value == 'F':
        Female += 1
    else:
        Male += 1

y = np.array([Male, Female])
my_labels = ["Male", "Female"]
my_colors = ["#1679AB", "#D24545"]


def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)


fig, ax = plt.subplots(figsize=(30, 20))

plt.pie(y, labels=my_labels, colors=my_colors, autopct=lambda pct: func(pct, y), )
ax.set_title("Male / Female Repartition")

plt.show()

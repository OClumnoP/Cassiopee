import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Patient = pd.read_csv(r"C:\Users\bajou\Documents\MIMIC\mimic-iii-clinical-database\PATIENTS.csv")
Patient = Patient[['EXPIRE_FLAG']]

Dead = 0
Alive = 0


for i in range(Patient.shape[0]):
    value = Patient.iat[i, 0]
    if value == 1:
        Dead += 1
    else:
        Alive += 1

y = np.array([Dead, Alive])
my_labels = ["Dead", "Alive"]
my_colors = ["#D24545", "#1679AB"]


def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)


fig, ax = plt.subplots(figsize=(7, 5))

plt.pie(y, labels=my_labels, colors=my_colors, autopct=lambda pct: func(pct, y), )
ax.set_title("Death Repartition")

plt.show()

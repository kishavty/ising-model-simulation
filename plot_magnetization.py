from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


Ts1 = np.concatenate((np.arange(0.5, 2.3, 0.2), np.arange(2.1, 2.55, 0.1), np.arange(2.5, 3.6, 0.2)))

Ts2 = np.concatenate((np.arange(0.5, 2.3, 0.2), np.arange(2.1, 2.55, 0.05), np.arange(2.5, 3.6, 0.2)))


M_L10 = pd.read_csv("csvmagnet/M_L10.csv")
M_L20 = pd.read_csv("csvmagnet/M_L20.csv")
M_L40 = pd.read_csv("csvmagnet/M_L40.csv")
M_L80 = pd.read_csv("csvmagnet/M_L80.csv")


norm10 = preprocessing.normalize([M_L10["m"]])
norm20 = preprocessing.normalize([M_L20["m"]])
norm40 = preprocessing.normalize([M_L40["m"]])
norm80 = preprocessing.normalize([M_L80["m"]])


"""
plt.plot(Ts1, M_L10['m'], marker='o', label="L = 10")
plt.plot(Ts1, M_L20['m'], marker='*', label="L = 20")
plt.plot(Ts2, M_L40['m'], marker='x', label="L = 40")
plt.plot(Ts2, M_L80['m'], marker='d', label="L = 80")"""

plt.plot(Ts1, norm10[0]*10/3, marker='o', label="L = 10")
plt.plot(Ts1, norm20[0]*10/3, marker='*', label="L = 20")
plt.plot(Ts2, norm40[0]*10/3, marker='x', label="L = 40")
plt.plot(Ts2, norm80[0]*10/3, marker='d', label="L = 80")

plt.xlabel("T")
plt.ylabel("<m>")
plt.legend()
plt.title("Magnetyzacja")
plt.savefig("magnety/m.png", dpi=500)
plt.show()
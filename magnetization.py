from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ising(L, T, mc_steps, random=False):
    N = L**2
    #kB = 1.380649 * 10**(-23) zał. kB=1
    m = []

    if random: #losowy
        table = np.random.choice([-1, 1], size=(L, L))
    else: #uporzadkowany
        table = np.ones((L, L))

    for i in tqdm(range(mc_steps)):
        for spin in range(N):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            #plt.savefig(f'l10t{T}mcs0.png')
            delta_E = 2 * table[x, y] * (table[(x-1)%L, y] + table[(x+1)%L, y] + table[x, (y-1)%L] + table[x, (y+1)%L])
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E/(T)): #kB = 1
                table[x, y] *= -1

        m.append(np.sum(table) / N)
        
    return m

def magnetization(L, T, mc_steps, m):
    K = 100000 - 10000
    #K = mc_steps - 10000
    for t in T:
        ms = ising(L, t, mc_steps, random=True)#[100:]
        m_ = np.sum(np.abs(ms)) / K
        m.append(m_)
        #X.append(L**2 / t * (np.sum(np.square(ms)) / K - m_**2))
    print(f"T = {t} | ", end="")
    return m

"""
### L=10
Ts = np.arange(0.5, 2.3, 0.2)
m = []
M_10 = magnetization(10, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L10_1.csv", index=False)

Ts = np.arange(2.1, 2.55, 0.1)
m = []
M_10 = magnetization(10, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L10_2.csv", index=False)

Ts = np.arange(2.5, 3.5, 0.2)
m = []
M_10 = magnetization(10, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L10_3.csv", index=False)

M1_L10 = pd.read_csv("csvmagnet/M_L10_1.csv")
M2_L10 = pd.read_csv("csvmagnet/M_L10_2.csv")
M3_L10 = pd.read_csv("csvmagnet/M_L10_3.csv")
m_L10 = np.concatenate((M1_L10['m'].values, M2_L10['m'].values, M3_L10['m'].values))
data = pd.DataFrame({'m': m_L10})#, 'X': X})
data.to_csv("csvmagnet/M_L10.csv", index=False)                 
M_L10 = pd.read_csv("csvmagnet/M_L10.csv")
"""

### L=20
Ts = np.arange(0.5, 2.3, 0.2)
m = []
M_20 = magnetization(20, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L20_1.csv", index=False)

Ts = np.arange(2.1, 2.55, 0.1)
m = []
M_20 = magnetization(20, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L20_2.csv", index=False)

Ts = np.arange(2.5, 3.7, 0.2)
m = []
M_20 = magnetization(20, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L20_3.csv", index=False)

M1_L20 = pd.read_csv("csvmagnet/M_L20_1.csv")
M2_L20 = pd.read_csv("csvmagnet/M_L20_2.csv")
M3_L20 = pd.read_csv("csvmagnet/M_L20_3.csv")
m_L20 = np.concatenate((M1_L20['m'].values, M2_L20['m'].values, M3_L20['m'].values))
data = pd.DataFrame({'m': m_L20})#, 'X': X})
data.to_csv("csvmagnet/M_L20.csv", index=False)                 
M_L20 = pd.read_csv("csvmagnet/M_L20.csv")

### L=80 
"""
Ts = np.arange(0.5, 2.3, 0.2)
m = []
M_80 = magnetization(80, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L80_1.csv", index=False)

Ts = np.arange(2.1, 2.55, 0.05)
m = []
M_80 = magnetization(80, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L80_2.csv", index=False)

Ts = np.arange(2.5, 3.7, 0.2)
m = []
M_80 = magnetization(80, Ts, 10000, m)#), X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L80_3.csv", index=False)

M1_L80 = pd.read_csv("csvmagnet/M_L80_1.csv")
M2_L80 = pd.read_csv("csvmagnet/M_L80_2.csv")
M3_L80 = pd.read_csv("csvmagnet/M_L80_3.csv")
m_L80 = np.concatenate((M1_L80['m'].values, M2_L80['m'].values, M3_L80['m'].values))
data = pd.DataFrame({'m': m_L80})#, 'X': X})
data.to_csv("csvmagnet/M_L80.csv", index=False)                 
M_L80 = pd.read_csv("csvmagnet/M_L80.csv")
"""
"""
### L=40
Ts = np.arange(0.5, 3.6, 0.1)
m = []
M_40 = magnetization(40, Ts, 1000000, m)#, X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L40.csv", index=False)

### L=100
Ts = np.arange(0.5, 2.1, 0.1)
m = []
#X = []
M_100_p1 = magnetization(100, Ts, 1000000, m)#, X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L100_p1.csv", index=False)

Ts = np.arange(2.05, 2.5, 0.05)
m = []
#X = []
M_40_p2 = magnetization(40, Ts, 1000000)#, m, X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L40_p2.csv", index=False)

Ts = np.arange(2.5, 3.6, 0.1)
m = []
#X = []
M_100_p3 = magnetization(100, Ts, 1000000)#, m, X)
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L100_p3.csv", index=False)


M1 = pd.read_csv("csvmagnet/M_L40.csv")
M2 = pd.read_csv("csvmagnet/M_L40_p2.csv")

m = np.concatenate((M1['m'].values[:11], M2['m'].values, M1['m'].values[16:]))
#X = np.concatenate((M1['X'].values[:11], M2['X'].values, M1['X'].values[16:]))
data = pd.DataFrame({'m': m})#, 'X': X})
data.to_csv("csvmagnet/M_L40_x.csv", index=False)

M_L10 = pd.read_csv("csvmagnet/M_L10.csv")
M_L50 = pd.read_csv("csvmagnet/M_L40.csv")
M_L100 = pd.read_csv("csvmagnet/M_L100.csv")
"""


Ts1 = np.concatenate((np.arange(0.5, 2.3, 0.2), np.arange(2.1, 2.55, 0.1), np.arange(2.5, 3.7, 0.2)))

Ts2 = np.concatenate((np.arange(0.5, 2.3, 0.2), np.arange(2.1, 2.55, 0.05), np.arange(2.5, 3.7, 0.2)))


#plt.plot(Ts1, M_L10['m'], marker='o', label="L = 10")
plt.plot(Ts1, M_L20['m'], marker='o', label="L = 20")
#plt.plot(Ts2, M_L80['m'], marker='o', label="L = 80")
#plt.plot(Ts1, M_L100['m'], marker='o', label="L = 100")
"""
plt.xlabel("T")
plt.ylabel("<m>")
plt.legend()
plt.title("Magnetyzacja")
plt.savefig("magnety/m80.png", dpi=500)
plt.show()
"""
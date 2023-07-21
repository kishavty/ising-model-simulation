import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas
from tqdm import tqdm
import time

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


for L in [80]:
    #L = 10
    T = 2.26
    mc_steps = 10000


    data = pandas.DataFrame()

    for i in range(1, 11):
        m = ising(L, T, mc_steps, True)
        data[f"m{i}"] = m

    data.to_csv(f"csvkigwiazdka/T{T}_L{L}.csv")

    plt.figure()
    data = pandas.read_csv(f"csvkigwiazdka/T{T}_L{L}.csv")

    for i in range(1, 11):
        m = data[f"m{i}"]
        plt.plot(m)

    plt.xlabel("t [MCS]")
    plt.ylabel("m")
    plt.title(f"Trajektorie dla T = {T}, L = {L}")
    plt.legend()
    plt.savefig(f"trajektoriegwiazdka/T{T}_L{L}.png", dpi=500)
    time.sleep(5)
    continue
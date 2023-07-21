import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

def ising(L, T, random=False):
    N = L**2
    #kB = 1.380649 * 10**(-23) zał. kB=1
    m = []

    if random: #losowy
        table = np.random.choice([-1, 1], size=(L, L))
    else: #uporzadkowany
        table = np.ones((L, L))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_title(f"T = {T}, MCS = 0")
    ax.axis('off')

    heatmap = ax.imshow(table, cmap='YlGnBu')

    def update(step):
        nonlocal table
        nonlocal m

        for spin in range(N):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            #plt.savefig(f'l10t{T}mcs0.png')
            delta_E = 2 * table[x, y] * (table[(x-1)%L, y] + table[(x+1)%L, y] + table[x, (y-1)%L] + table[x, (y+1)%L])
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E/(T)): #kB = 1
                table[x, y] *= -1

        m.append(np.sum(table) / N)

        heatmap.set_data(table)
        ax.set_title(f"T = {T}, MCS = {step}")
        if step == 100:
            plt.savefig(f'l{L}t{T}mcs{step}.png')
            quit()

    animation = FuncAnimation(fig, update, frames=100, interval=10)

    plt.show()
    return m

m = ising(L=10, T=4, random=False)

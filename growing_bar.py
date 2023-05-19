import matplotlib.pyplot as plt
import time

fig, ax = plt.subplots()
ax.set_ylim(0, 10)
bars = ax.bar(['a', 'b', 'c'], 0, width=1, color='red')
for i in range(10):
    bars[0].set_height(i)
    bars[1].set_height(i*2)
    plt.pause(1)

plt.show()
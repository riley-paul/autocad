import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from random import random
from space_ranges import space_ranges

domain = 10

data = pd.DataFrame()
data["beg"] = sorted([random() * domain for _ in range(15)])
data["end"] = data.beg.apply(lambda i: i + random() * 3)
data["color"] = [[random() * 0.8 for _ in range(3)] + [0.5] for _ in range(len(data))]
data["height"] = [random() for _ in range(len(data))]

ranges = list(zip(data.beg, data.end))
ranges = space_ranges(ranges, 0.2, iterations=100)

data["beg_spaced"] = [i[0] for i in ranges]
data["end_spaced"] = [i[1] for i in ranges]

## PLOT ##
fig, (ax1, ax2) = plt.subplots(2, sharex=True)


def plot_value(beg, end, ax, height, color):
    x = [beg, beg, end, end]
    y = [0, height, height, 0]
    ax.plot(x, y, color=color)


data.apply(lambda r: plot_value(r.beg, r.end, ax1, r.height, r.color), axis=1)
data.apply(
    lambda r: plot_value(r.beg_spaced, r.end_spaced, ax2, r.height, r.color), axis=1
)

plt.show()

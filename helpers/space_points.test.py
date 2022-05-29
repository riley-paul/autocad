import matplotlib.pyplot as plt
import matplotlib.patches as patches

from random import random
from space_points import space_ranges,space_points
import pandas as pd

def plot_ranges(ranges,ax):
  for index,row in ranges.iterrows():
    height = row.height + 1
    
    x = [row.beg,row.beg,row.end,row.end]
    y = [0,height,height,0]
    ax.plot(x,y)
  ax.set_xlim(0,120)

domain = 100

# generate random data
ranges = pd.DataFrame()
ranges["beg"] = [random() * domain for _ in range(15)]
ranges["end"] = ranges.beg.apply(lambda i: i + random() * 5)
ranges["height"] = ranges.index
ranges = ranges.sort_values("beg").reset_index(drop=True)

# space out data using algorithm
# spaced = space_ranges(ranges.apply(lambda r: [r.beg,r.end],axis=1).tolist())
# ranges_spaced = pd.DataFrame()
# ranges_spaced["beg"] = [i[0] for i in spaced]
# ranges_spaced["end"] = [i[1] for i in spaced]

ranges_spaced = space_ranges(ranges)

# Plot results
fig,(ax1,ax2) = plt.subplots(2)
plot_ranges(ranges,ax1)
plot_ranges(ranges_spaced,ax2)

plt.show()
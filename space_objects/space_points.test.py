import matplotlib.pyplot as plt
import pandas as pd

from random import random
from space_points import space_points

domain = 10

data = pd.DataFrame()
data["value"] = sorted([random() * domain for _ in range(7)])
data["value_spaced"] = space_points(data["value"].tolist(),1)
data["color"] = [[random() for _ in range(3)] for _ in range(len(data))]
print(data)

## PLOT ##
fig,(ax1,ax2) = plt.subplots(2)

def plot_value(value,ax,color = [0,0,255]):
  ax.plot([value],[0],"o",color=color)

data.apply(lambda r: plot_value(r.value,ax1,r.color),axis=1)
data.apply(lambda r: plot_value(r.value_spaced,ax2,r.color),axis=1)

plt.show()
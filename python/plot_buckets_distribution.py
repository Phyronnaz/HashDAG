import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog

initialdir = os.path.realpath(os.path.realpath(__file__) + "/../../profiling/")
print(initialdir)

path = filedialog.askopenfile(filetypes=[("CSV files", "*.csv")],
                              initialdir=initialdir)

name = os.path.splitext(os.path.basename(path.name))[0]

print("Loading " + name + "... ", end="")

# panda is WAY faster than numpy
data = pd.read_csv(path.name, header=None, delimiter=";", dtype='unicode').values
print("Loaded")

level_buckets = {}
level = -1
for line in data:
    assert len(line) == 2, "line: {}".format(line)
    if line[0] == "level":
        level = int(line[1])
        level_buckets[level] = []
    else:
        level_buckets[level].append(int(line[1]))

plot_index = 0
for level, buckets in level_buckets.items():
    if max(buckets) == 0:
        continue
    plot_index += 1
    plt.subplot(4, 4, plot_index)
    plt.title("level = " + str(level))
    buckets = np.array(buckets)
    plt.hist(buckets, color="blue", bins=range(int(np.min(buckets)), int(np.max(buckets) + 1)))
plt.show()

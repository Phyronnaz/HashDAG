import numpy as np

import tools
from tools import get_array, dags, results_prompt, get_custom_array
import matplotlib.pyplot as plt
#import seaborn as sns

path = results_prompt("memory")

# Styling
plt.style.use("seaborn")

#plt.rc("font", family="serif", size=56)
#plt.rc("axes", labelsize=18)
#plt.rc("legend", fontsize=12)

plt.rc("font", family="serif", size=14)
plt.rc("axes", labelsize=32)
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
plt.rc("legend", fontsize=24)

# Setup
fig = plt.figure(dpi=120, figsize=(10, 6))
ax = fig.add_subplot(111)


# Plot
data = dags[0]

plt.xlabel("Cumulative #edited voxels")
plt.ylabel("Memory usage (MB)")

kwargs = {"marker": "", "markersize": 5}

num_voxels = get_array(data, "num voxels")
voxels_indices = np.cumsum(num_voxels)

def get_custom_array(name):
    freed_memory = np.zeros(len(tools.indices))
    for i in range(len(tools.indices)):
        index = tools.indices[i]
        d = data[data[:, 0].astype(np.int) == index]
        freed_memory[i] = float(d[d[:, 1] == name][0][2])
    return freed_memory

# Plot
virtual_size = get_custom_array("virtual_size")
ax.plot(voxels_indices, virtual_size, label="Total size", linewidth=5)

gc_freed_memory = get_custom_array("GC freed memory leaf level")
ax.plot(voxels_indices, virtual_size - gc_freed_memory, label="Total size after full GC", linewidth=5)

selected = [10, 11, 12, 13];#, 14];
#depth = 16
#for i in range(depth - 1):
for j in range(len(selected)):
    i = selected[j];
    gc_freed_memory = get_custom_array("GC freed memory level " + str(i))
    if j == 1:
        ax.plot(voxels_indices, virtual_size - gc_freed_memory, color = 'C5', label="Up to level " + str(i), **kwargs)
    else:
        ax.plot(voxels_indices, virtual_size - gc_freed_memory, label="Up to level " + str(i), **kwargs)

ax.ticklabel_format( axis='x', style='sci', scilimits=(0,0) );
leg = plt.legend()
#for legobj in leg.legendHandles:
#    legobj.set_linewidth(6.0)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
plt.savefig(path + "plot.pdf", format="pdf")
plt.show()

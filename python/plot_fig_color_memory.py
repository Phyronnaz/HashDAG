import numpy as np

from tools import get_array, dags, results_prompt, get_custom_array
import matplotlib.pyplot as plt

path = results_prompt("color_memory")

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

color_size = get_custom_array(data, "color_size")
undo_redo_size = get_custom_array(data, "color_size undo_redo")

ax.plot(voxels_indices, color_size + undo_redo_size, label="With history", **kwargs)
ax.plot(voxels_indices, color_size, label="Without history", **kwargs)


ax.ticklabel_format( axis='x', style='sci', scilimits=(0,0) );
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(6.0)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
plt.savefig(path + "plot.pdf", format="pdf")
plt.show()

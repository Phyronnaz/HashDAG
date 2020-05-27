import numpy as np

from tools import get_array, dags, profiling_prompt, get_custom_array
import matplotlib.pyplot as plt

profiling_prompt()

data = dags[0]

plt.xlabel("cumulative num voxels")
plt.ylabel("Memory usage (MB)")

kwargs = {"marker": "", "markersize": 5}

num_voxels = get_array(data, "num voxels")
voxels_indices = np.cumsum(num_voxels)

virtual_size = get_custom_array(data, "virtual_size")
plt.plot(voxels_indices, virtual_size, label="Virtual size", **kwargs)

gc_freed_memory = get_custom_array(data, "GC freed memory leaf level")
plt.plot(voxels_indices, virtual_size - gc_freed_memory, label="Virtual size after GC", marker="x")

depth = 16
for i in range(depth - 1):
    gc_freed_memory = get_custom_array(data, "GC freed memory level " + str(i))
    plt.plot(voxels_indices, virtual_size - gc_freed_memory, label="Virtual size after GC level " + str(i), **kwargs)

plt.legend()
plt.show()

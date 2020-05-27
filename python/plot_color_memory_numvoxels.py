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

color_size = get_custom_array(data, "color_size")
undo_redo_size = get_custom_array(data, "color_size undo_redo")

plt.plot(voxels_indices, color_size + undo_redo_size, label="Color size with undo redo history", **kwargs)
plt.plot(voxels_indices, color_size, label="Color size without undo redo history", **kwargs)

plt.legend()
plt.show()

import tools
from tools import get_arrays_median, single_dags, dags
import matplotlib.pyplot as plt

kwargs = {"marker": "", "markersize": 2}

single_dags_paths = get_arrays_median(single_dags, "paths")
single_dags_colors = get_arrays_median(single_dags, "colors")
hash_dags_paths = get_arrays_median(dags, "paths")
hash_dags_colors = get_arrays_median(dags, "colors")

plt.xlabel("Frame")
plt.ylabel("Time (ms)")

# plt.xlim(0, 1000)
# plt.ylim(0, 5)

plt.plot(tools.indices, single_dags_paths, color="red", label="paths (DAG)", **kwargs)
plt.plot(tools.indices, single_dags_colors, color="orange", label="colors (DAG)", **kwargs)
plt.plot(tools.indices, hash_dags_paths, color="green", label="paths (Hash DAG)", **kwargs)
plt.plot(tools.indices, hash_dags_colors, color="cyan", label="colors (Hash DAG)", **kwargs)

plt.legend()
plt.show()

# plt.plot(tools.indices, (hash_dags_paths - single_dags_paths) / single_dags_paths, color="blue", **kwargs)
# plt.plot(tools.indices, (hash_dags_colors - single_dags_colors) / single_dags_colors, color="purple", **kwargs)
# plt.show()
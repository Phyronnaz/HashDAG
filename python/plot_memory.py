import tools
from tools import get_array, dags, profiling_prompt
import matplotlib.pyplot as plt

profiling_prompt()

data = dags[0]

virtual_size = get_array(data, "virtual_size")

plt.xlabel("Frame")
plt.ylabel("Memory usage (MB)")

kwargs = {"marker": "", "markersize": 5}

plt.plot(tools.indices, virtual_size, label="Virtual size", **kwargs)

gc_freed_memory = get_array(data, "GC freed memory leaf level")
plt.plot(tools.indices, virtual_size - gc_freed_memory, label="Virtual size after GC", marker="x")

depth = 16
for i in range(depth - 1):
    gc_freed_memory = get_array(data, "GC freed memory level " + str(i))
    plt.plot(tools.indices, virtual_size - gc_freed_memory, label="Virtual size after GC level " + str(i), **kwargs)

plt.legend()
plt.show()

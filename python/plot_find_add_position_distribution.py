import numpy as np
import matplotlib.pyplot as plt
from tools import get_array, dags

data = dags[0]

find_or_add_index = get_array(data, "find_or_add index")
find_or_add_add = get_array(data, "find_or_add add")
find_or_add_level = get_array(data, "find_or_add level")

plt.xlabel("index")
plt.ylabel("num")

indices_add = find_or_add_add == 1
indices_noadd = find_or_add_add == 0

print("Num add: ", np.sum(indices_add))
print("Num no add: ", np.sum(indices_noadd))

assert np.sum(indices_add) + np.sum(indices_noadd) == len(find_or_add_index)

plot_index = 0
for level in range(32):
    level_indices = level == find_or_add_level
    if np.sum(level_indices) == 0:
        continue

    plot_index += 1
    plt.subplot(2, 3, plot_index)
    plt.title("level = " + str(level))
    indices = find_or_add_index[np.logical_and(indices_noadd, level_indices)]
    plt.hist(indices, color="blue", bins=range(0, int(np.max(indices) + 1)))
plt.show()

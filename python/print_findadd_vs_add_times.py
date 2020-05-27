import numpy as np
from tools import get_array, dags

data = dags[0]

find_or_add_index = get_array(data, "find_or_add index")
find_or_add_time = get_array(data, "find_or_add time")
find_or_add_add = get_array(data, "find_or_add add")
find_or_add_level = get_array(data, "find_or_add level")

indices_add = find_or_add_add == 1
indices_noadd = find_or_add_add == 0

print("Num add: ", np.sum(indices_add))
print("Num no add: ", np.sum(indices_noadd))

assert np.sum(indices_add) + np.sum(indices_noadd) == len(find_or_add_index)

for level in range(32):
    level_indices = level == find_or_add_level
    if np.sum(level_indices) == 0:
        continue

    print("level {}: mean no add: {}; mean add: {}; sum no add: {}; sum add: {}".format(
        level,
        np.mean(find_or_add_time[np.logical_and(indices_noadd, level_indices)]),
        np.mean(find_or_add_time[np.logical_and(indices_add, level_indices)]),
        np.sum(find_or_add_time[np.logical_and(indices_noadd, level_indices)]),
        np.sum(find_or_add_time[np.logical_and(indices_add, level_indices)])))

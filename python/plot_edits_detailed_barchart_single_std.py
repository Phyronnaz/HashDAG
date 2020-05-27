import numpy as np
import matplotlib.pyplot as plt
from tools import get_array, dags, names

plots_map = {}
names = ["first pass", "start threads", "waiting", "second pass"]
names = ["first pass", "leaf level edits", "high level edits", "bunch color copy", "full color copy", "find_node",
         "add_node", "second pass"]
# names = ["leaf level edits average", "high level edits average", "bunch color copy average", "find_or_add average"]
# names = ["find_or_add average"]

num_nodes = None
for data in dags:
    new_num_nodes = get_array(data, "num nodes")
    assert num_nodes is None or np.all(num_nodes == new_num_nodes), "Edits are different!"
    num_nodes = new_num_nodes

    for name in names:
        if name not in plots_map:
            plots_map[name] = []
        plots_map[name].append(get_array(data, name))

width = 0.3

indices = np.argsort(num_nodes)
plt.xlabel("nodes")
plt.ylabel("time (ms)")

if len(names) == 1:
    plt.title(names[0])
# plt.ylim(0, 250)

previous = np.zeros(len(indices))
for (name, plots) in plots_map.items():
    data = np.mean(plots, axis=0)[indices]
    plt.bar(indices, data, yerr=np.std(plots, axis=0)[indices], bottom=previous, width=width, label=name)
    previous += data

plt.xticks(indices, num_nodes[indices])

plt.legend()
plt.show()

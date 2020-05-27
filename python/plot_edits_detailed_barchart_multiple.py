import numpy as np
import matplotlib.pyplot as plt
from tools import get_array, dags, names, profiling_prompt

profiling_prompt()

plots_map = {}
names = [
    "first pass",
    "second pass",
    "early exit checks",
    "leaf edit",
    "find or add leaf",
    "interior edit",
    "find or add interior",
    # "start threads",
    # "waiting",
    "entirely full - add colors",
    "skip edit - copy colors"]

num_nodes = None
total_num_nodes = []
for data in dags:
    new_num_nodes = get_array(data, "num voxels")
    assert num_nodes is None or np.all(num_nodes == new_num_nodes), "Edits are different!"
    num_nodes = new_num_nodes

    total_num_nodes += list(num_nodes)

    for name in names:
        if name not in plots_map:
            plots_map[name] = []
        plots_map[name] += list(get_array(data, name))

total_num_nodes = np.array(total_num_nodes)

width = 0.5

indices = np.argsort(total_num_nodes, kind="stable")
plt.xlabel("num voxels")
plt.ylabel("time (ms)")

if len(names) == 1:
    plt.title(names[0])
# plt.ylim(0, 250)

assert len(dags) * len(num_nodes) == len(total_num_nodes)

plot_indices = np.array(
    [len(dags) * (i // len(dags)) + (i % len(dags)) * (width + 0.05) for i in range(len(dags) * len(num_nodes))])
previous = np.zeros(len(indices))
for name, plots in plots_map.items():
    plots = np.array(plots)[indices]
    plt.bar(plot_indices, plots, bottom=previous, width=width, label=name)
    previous += plots

plt.xticks(len(dags) * np.array(range(len(num_nodes))), np.sort(num_nodes))

plt.legend()
plt.show()

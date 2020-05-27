import matplotlib
import numpy as np
from tools import get_array, dags, page_sizes
import matplotlib.pyplot as plt

edits_time = []
edits_num_nodes = []
for dag in dags:
    edits_time.append(get_array(dag, "edits"))
    edits_num_nodes.append(get_array(dag, "numNodes"))

kwargs = {"linestyle": "None", "marker": "o", "markersize": 5}

plt.xlabel("nodes")
plt.ylabel("time")

cmap = matplotlib.cm.get_cmap('Spectral')

log_page_sizes = np.log2(np.array(page_sizes))
log_page_sizes = (log_page_sizes - np.min(log_page_sizes)) / (np.max(log_page_sizes) - np.min(log_page_sizes))
labels = set()
for i in range(len(dags)):
    color = cmap(log_page_sizes[i])
    label = page_sizes[i]
    if label in labels:
        label = ""
    else:
        labels.add(label)
    plt.plot(edits_num_nodes[i], edits_time[i], color=color, label=label, **kwargs)

plt.legend()
plt.show()

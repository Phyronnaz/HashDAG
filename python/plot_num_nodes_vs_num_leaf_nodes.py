import tools
from tools import get_array, dags
import matplotlib.pyplot as plt

data = dags[0]

edits_num_nodes = get_array(data, "num nodes")
edits_num_leaf_nodes = get_array(data, "num leaf nodes")

kwargs = {"linestyle": "None", "marker": "o", "markersize": 5}

plt.xlabel("frame")
plt.ylabel("nodes")
plt.plot(tools.indices, edits_num_nodes, label="num nodes", **kwargs)
plt.plot(tools.indices, edits_num_leaf_nodes, label="num leaf nodes", **kwargs)
plt.legend();
plt.show()

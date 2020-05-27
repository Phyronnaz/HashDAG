from tools import get_array, dags
import matplotlib.pyplot as plt

data = dags[0]

edits_time = get_array(data, "edits")
edits_num_nodes = get_array(data, "num nodes")

kwargs = {"linestyle": "None", "marker": "o", "markersize": 5}

plt.xlabel("nodes")
plt.ylabel("time")
plt.plot(edits_num_nodes, edits_time, **kwargs)

if "rebuilding colors" in data[:, 1]:
    colors_time = get_array(data, "rebuilding colors")
    print("plotting rebuilding colors")
    plt.plot(edits_num_nodes, edits_time + colors_time, color="red", **kwargs)

plt.show()

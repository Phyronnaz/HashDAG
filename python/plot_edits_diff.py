from tools import get_array, dags, names, profiling_prompt
import matplotlib.pyplot as plt

profiling_prompt()

for i in range(len(dags)):
    data = dags[i]
    edits_time = get_array(data, "edits")
    edits_num_voxels = get_array(data, "num voxels")

    kwargs = {"linestyle": "None", "marker": "o", "markersize": 5}

    plt.xlabel("num voxels")
    plt.ylabel("time")
    plt.plot(edits_num_voxels, edits_time, label=names[i], **kwargs)

    #if "rebuildColorsTime" in data[:, 1]:
    #    colors_time = get_array(data, "rebuildColorsTime")
    #    plt.plot(edits_num_nodes, edits_time + colors_time, color="red", **kwargs)

plt.legend()
plt.show()

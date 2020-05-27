from tools import get_array, dags
import matplotlib.pyplot as plt

data = dags[0]

edits_time = get_array(data, "edits")
edits_radius = get_array(data, "radius")
edits_num_findadd = get_array(data, "num find or add")
edits_num_iteratedvoxels = get_array(data, "num iterated leaf voxels")
edits_num_editedvoxels = get_array(data, "num edited leaf voxels")
edits_num_nodes = get_array(data, "num nodes")

kwargs = {"linestyle": "None", "marker": "o", "markersize": 5}

plt.xlabel("radius")
plt.ylabel("time")
plt.plot(edits_radius, edits_time, **kwargs)
plt.figure()

plt.xlabel("findadd")
plt.ylabel("time")
plt.plot(edits_num_findadd, edits_time, **kwargs)
plt.figure()

plt.xlabel("Edited nodes")
plt.ylabel("Time (ms)")
plt.plot(edits_num_nodes, edits_time, **kwargs)
plt.figure()

plt.xlabel("iterated voxels")
plt.ylabel("time")
plt.plot(edits_num_iteratedvoxels, edits_time, **kwargs)
plt.figure()

plt.xlabel("edited voxels")
plt.ylabel("time")
plt.plot(edits_num_editedvoxels, edits_time, **kwargs)
plt.figure()

plt.xlabel("radius")
plt.ylabel("iterated voxels")
plt.plot(edits_radius, edits_num_iteratedvoxels, **kwargs)
plt.figure()

plt.xlabel("radius")
plt.ylabel("num edited nodes/num nodes")
plt.plot(edits_radius, edits_num_findadd / edits_num_nodes, **kwargs)
plt.figure()

plt.xlabel("Radius")
plt.ylabel("Edited Nodes")
plt.tight_layout(3)
plt.plot(edits_radius, edits_num_nodes, **kwargs)

plt.show()

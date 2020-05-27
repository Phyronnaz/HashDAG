import matplotlib.pyplot as plt
import numpy as np
import tools
from tools import get_array, dags, defines

path = tools.results_prompt("rt")

# Styling
plt.style.use("seaborn")

#plt.rc("font", family="serif", size=56)
#plt.rc("axes", labelsize=18)
#plt.rc("legend", fontsize=12)

plt.rc("font", family="serif", size=14)
plt.rc("axes", labelsize=32)
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
plt.rc("legend", fontsize=22)

# Setup
fig = plt.figure(dpi=120, figsize=(10, 8))
ax = fig.add_subplot(111)


# Labels
labels_geo = [
    "Geometry (HashDAG)",
    "Geometry (original)"
]
labels_col = [
    "Colors (HashDAG)",
    "Colors (original)"
]

times = [None, None]

# Do stuff
for i in range(len(dags)):
    data = dags[i]
    label_index = defines[i]["USE_NORMAL_DAG"] == "1"
    paths = get_array(data, "paths")
    colors = get_array(data, "colors")

    times[label_index] = paths + colors

    kwargs = {"marker": "", "markersize": 2}

    ax.set_xlabel("frame number")
    ax.set_ylabel("time (ms)")

    ax.plot(tools.indices, paths, label=labels_geo[label_index], **kwargs)
    ax.plot(tools.indices, colors, label=labels_col[label_index], **kwargs)

diff = times[0] / times[1]

print("ratio total ours/total original: avg: {}; std: {}; min: {}; max: {}".format(np.mean(diff), np.std(diff),
                                                                                   np.min(diff), np.max(diff)))

leg = ax.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(6.0)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
plt.savefig(path + "plot.pdf", format="pdf")
plt.show()

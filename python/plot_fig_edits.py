import matplotlib.pyplot as plt
import re;
import tools

from tools import get_array, dags, defines, names

path = tools.results_prompt("edits")

# Styling
plt.style.use("seaborn")

#plt.rc("font", family="serif", size=56)
#plt.rc("axes", labelsize=18)
#plt.rc("legend", fontsize=12)

plt.rc("font", family="serif", size=14)
plt.rc("axes", labelsize=32)
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
plt.rc("legend", fontsize=24)

# Setup
fig = plt.figure(dpi=100, figsize=(6, 10))
ax = fig.add_subplot(111)

# Do stuff
for i in range(len(dags)):
    data = dags[i]
    label_index = 0
    time = get_array(data, "edits")
    num_voxels = get_array(data, "num voxels")

    label = "(MISSING LABEL)";
    if -1 != names[i].find( "nothread" ):
        label = "Single-threaded";
    else:
        threadCount = re.search( r"_thread(\d+)", names[i] ).group(1);
        label = "{} threads".format( threadCount );

    kwargs = {"linestyle": "None", "marker": "o", "markersize": 5}

    ax.set_xlabel("edited voxels")
    ax.set_ylabel("time (ms)")

    ax.plot(num_voxels, time, label=label, **kwargs)

ax.ticklabel_format( axis='x', style='sci', scilimits=(0,0) );
leg = ax.legend( markerscale = 4 )

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
plt.savefig(path + "plot.pdf", format="pdf")
plt.show()

import matplotlib.pyplot as plt
import tools
from tools import get_array, dags, names, profiling_prompt

profiling_prompt()

plt.style.use("seaborn");

fig = plt.figure( dpi = 100, figsize=(6,8) );
ax = fig.add_subplot(111);

for i in range(len(dags)):
    data = dags[i]
    paths = get_array(data, "paths")
    colors = get_array(data, "colors")

    kwargs = {"marker": "", "markersize": 2}

    ax.set_xlabel("frames")
    ax.set_ylabel("time (ms)")

    #color = "red" if "headless=1" in names[i] else "green"
    #ax.plot(tools.indices, paths, label="paths " + names[i], color=color, **kwargs)
    #ax.plot(tools.indices, colors, label="colors " + names[i], color=color, **kwargs)
    ax.plot(tools.indices, paths, label="paths " + names[i],  **kwargs)
    ax.plot(tools.indices, colors, label="colors " + names[i],  **kwargs)

#ax.set_aspect( 1.0/ax.get_data_ratio()*4.0/3.0 );

leg = ax.legend()
#leg.set_in_layout(False);

plt.tight_layout( pad=0.4, w_pad=0.5, h_pad=1.0 );
plt.show()


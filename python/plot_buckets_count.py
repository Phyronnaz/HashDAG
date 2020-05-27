import matplotlib
import numpy as np
from tools import get_array, dags, buckets, get_arrays_median
import matplotlib.pyplot as plt

buckets_dags = {}
for i in range(len(dags)):
    bucket = buckets[i]

    if bucket not in buckets_dags:
        buckets_dags[bucket] = []

    buckets_dags[bucket].append(dags[i])

edits_time = {}
edits_num_nodes = {}
for (bucket, dags) in buckets_dags.items():
    edits_time[bucket] = get_arrays_median(dags, "edits")
    edits_num_nodes[bucket] = get_arrays_median(dags, "numNodes")


kwargs = {"linestyle": "None", "marker": "o", "markersize": 5}

plt.xlabel("Edited nodes")
plt.ylabel("Time (ms)")

cmap = matplotlib.cm.get_cmap('Spectral')

log_buckets = np.log2(np.array(buckets))
for bucket in reversed(sorted(set(buckets))):
    color = cmap((np.log2(bucket) - np.min(log_buckets)) / (np.max(log_buckets) - np.min(log_buckets)))
    plt.plot(edits_num_nodes[bucket], edits_time[bucket], color=color, label=str(bucket) + " buckets", **kwargs)

plt.legend()
plt.show()

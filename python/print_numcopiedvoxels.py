import numpy as np
from tools import get_array, dags, profiling_prompt

profiling_prompt()

data = dags[0]

num_voxels = get_array(data, "copied voxels")

print("{} copied voxels".format(np.sum(num_voxels)))
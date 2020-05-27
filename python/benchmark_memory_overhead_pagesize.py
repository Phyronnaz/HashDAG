import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "singleframe"

page_sizes = [1024, 128, 256, 512]

base_defines = [
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("REPLAY_NAME", "\"{}\"".format(replay_name)),
    ("ENABLE_CHECKS", "1"),  # nice to have and no impact as we are looking at the memory usage
    ("BUCKETS_BITS_FOR_LOW_LEVELS", "16")
]

path = get_result_path("page_size_memory_overhead")

for i in range(len(page_sizes)):
    ps = page_sizes[i]
    threaded = base_defines + [
        ("PAGE_SIZE", ps)
    ]
    run(threaded, "scene={}_depth={}_pagesize={}".format(scene, scene_depth, ps), path)

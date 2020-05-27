import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "edits"

page_sizes = [128, 256, 512];

threads = "1"; #"0"; # "1"

num_threads = "6";
base_defines = [
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("REPLAY_NAME", "\"{}\"".format(replay_name)),
    ("USE_BLOOM_FILTER", "0"),
    ("EDITS_COUNTERS", "1"),
    ("THREADED_EDITS", threads),
    ("NUM_THREADS", num_threads),
    ("EDITS_ENABLE_COLORS", "0"),
]


# At level 17, we need a bit more space in the buckets.
# This mainly increases the memory consumption of the page size
if scene_depth >= 17:
    base_defines += [
        ("BUCKETS_SIZE_FOR_LOW_LEVELS", 2048+1024)
    ]

path = get_result_path("page_size")

for i in range(len(page_sizes)):
    ps = page_sizes[i]
    threaded=base_defines + [
        ("PAGE_SIZE", ps),
    ];
    run(threaded, "scene={}_depth={}_ps{}".format(scene, scene_depth, ps), path)
pass;

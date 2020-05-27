import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "edits"

bucket_bits = [20, 18, 16]
bucket_sizes = [256, 1024, 2048];

# At level 17, we need a bit more space in the buckets.
# This mainly increases the memory consumption of the page size
#
# Note: 
if scene_depth >= 17:
    bucket_bits = [18, 17, 16]
    bucket_sizes = [1024+512, 2048, 2048+1024];

threads = "1";

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
    ("PAGE_SIZE", "128"),
    #("BUCKETS_BITS_FOR_TOP_LEVELS", "9")
]


path = get_result_path("bucket_count")

for i in range(len(bucket_bits)):
    bb = bucket_bits[i]
    bs = bucket_sizes[i]
    threaded=base_defines + [
        ("BUCKETS_BITS_FOR_LOW_LEVELS", bb),
        ("BUCKETS_SIZE_FOR_LOW_LEVELS", bs)
    ];
    run(threaded, "scene={}_depth={}_bbits{}_bsize{}".format(scene, scene_depth, bb,bs), path)
pass;

import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "copy"

base_defines = [
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("REPLAY_NAME", "\"{}\"".format(replay_name)),
    ("USE_BLOOM_FILTER", "0"),
    ("EDITS_COUNTERS", "1"),
    #("EDITS_PROFILING", "0"),
    ("USE_VIDEO", "0"),
    ("COPY_APPLY_TRANSFORM", "0"),
    ("COPY_CAN_APPLY_SWIRL", "0"),
    ("VERBOSE_EDIT_TIMES", "0"),
    ("TRACK_GLOBAL_NEWDELETE", "0"), # XXX
    ("THREADED_EDITS", "1"),
    ("NUM_THREADS", "6"),
    ("TRACY_ENABLE", "0"),
    #("ENABLE_CHECKS", "1"), # XXX XXX XXX XXX XXX
]

# At level 17, we need a bit more space in the buckets.
# This mainly increases the memory consumption of the page size
if scene_depth >= 17:
    #base_defines += [ ("BUCKETS_SIZE_FOR_LOW_LEVELS", (2*2048)) ];
    #base_defines += [ ("BUCKETS_SIZE_FOR_TOP_LEVELS", (2048)) ];
    pass;

path = get_result_path("copy_compdecomp")

normal6 = base_defines + [
    ("COPY_WITHOUT_DECOMPRESSION", "0"),
    ("COPY_EMPTY_CHECKS", "0"),
]
run(normal6, "scene={}_depth={}_t6_normal".format(scene, scene_depth), path)

comp6 = base_defines + [
    ("COPY_WITHOUT_DECOMPRESSION", "1"),
    ("COPY_EMPTY_CHECKS", "0"), 
]
run(comp6, "scene={}_depth={}_t6_comp".format(scene, scene_depth), path)

comp6e = base_defines + [
    ("COPY_WITHOUT_DECOMPRESSION", "1"),
    ("COPY_EMPTY_CHECKS", "1"),
]
run(comp6e, "scene={}_depth={}_t6_compempty".format(scene, scene_depth), path)

comp6e = base_defines + [
    ("COPY_WITHOUT_DECOMPRESSION", "0"),
    ("COPY_EMPTY_CHECKS", "1"),
]
run(comp6e, "scene={}_depth={}_t6_decempty".format(scene, scene_depth), path)


import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "tree_copy2"

base_defines = [
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("REPLAY_NAME", "\"{}\"".format(replay_name)),
    ("USE_BLOOM_FILTER", "0"),
    ("EDITS_COUNTERS", "1"),
    ("COUNT_COPIED_VOXELS", "1"),
    ("THREADED_EDITS", "1"),
    ("NUM_THREADS", 6),
    ("USE_VIDEO", "0"),
    ("COPY_APPLY_TRANSFORM", "0"),
    ("COPY_CAN_APPLY_SWIRL", "0"),
    ("VERBOSE_EDIT_TIMES", "0"),
    ("TRACK_GLOBAL_NEWDELETE", "0"), # XXX
    ("TRACY_ENABLE", "0"),
    ("COPY_WITHOUT_DECOMPRESSION", "0"),
    ("COPY_EMPTY_CHECKS", "0"),
]

path = get_result_path("copy_edit_voxels_sm")

run(base_defines, "scene={}_depth={}".format(scene, scene_depth), path)

#EOF

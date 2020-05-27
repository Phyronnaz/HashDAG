import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_names = ["teaser_statue", "teaser_church"];

base_defines = [
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("USE_BLOOM_FILTER", "0"),
    ("EDITS_COUNTERS", "1"),
    ("COUNT_COPIED_VOXELS", "1"),
    ("THREADED_EDITS", "1"),
    ("REPLAY_DEPTH", "17"),
    ("NUM_THREADS", 6)
]

if scene_depth >= 17:
    base_defines += [ ("BUCKETS_SIZE_FOR_LOW_LEVELS", (2048+1024) ) ];

path = get_result_path("teaser_voxels")

for i in range(len(replay_names)):
    name = replay_names[i];
    defs = base_defines + [
        ("REPLAY_NAME", "\"{}\"".format(name))
    ];
    run(defs, "scene={}_depth={}_{}".format(scene, scene_depth,name), path)
pass;

#EOF

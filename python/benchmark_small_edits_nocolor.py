import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "small_edits"

num_threads = ["2", "4", "6"];
base_defines = [
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("REPLAY_NAME", "\"{}\"".format(replay_name)),
    ("USE_BLOOM_FILTER", "0"),
    ("EDITS_ENABLE_COLORS", "0"),
    ("EDITS_COUNTERS", "1"),
]

path = get_result_path("edits")

unthreaded=base_defines + [
    ("THREADED_EDITS", "0")
];
run(unthreaded, "scene={}_depth={}_nothread".format(scene, scene_depth), path)

for i in range(len(num_threads)):
    nt = num_threads[i]
    threaded=base_defines + [
        ("THREADED_EDITS", "1"),
        ("NUM_THREADS", num_threads[i])
    ];
    run(threaded, "scene={}_depth={}_thread{}".format(scene, scene_depth, num_threads[i]), path)
pass;

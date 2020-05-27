import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "edits"

base_defines = [
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("REPLAY_NAME", "\"{}\"".format(replay_name)),
    ("EDITS_COUNTERS", "1"),
    ("UNDO_REDO", "1"),
]

path = get_result_path("color_memory")

run(base_defines, "scene={}_depth={}".format(scene, scene_depth), path)

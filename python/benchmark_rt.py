import sys

from benchmark_paths import get_result_path
from benchmark_tools import run

assert len(sys.argv) == 3, "Usage: python script.py scene_name scene_depth"

scene = sys.argv[1]
scene_depth = int(sys.argv[2])
replay_name = "move"

base_defines = [
    ("REPLAY_TWICE", "1"),
    ("ENABLE_SHADOWS", "0"),
    ("SCENE", "\"{}\"".format(scene)),
    ("SCENE_DEPTH", "{}".format(scene_depth)),
    ("REPLAY_NAME", "\"{}\"".format(replay_name)),
]

path = get_result_path("rt")

run(base_defines + [("USE_NORMAL_DAG", "1")], "scene={}_depth={}_normal_dag".format(scene, scene_depth), path)
run(base_defines + [("USE_NORMAL_DAG", "0")], "scene={}_depth={}_hash_dag".format(scene, scene_depth), path)

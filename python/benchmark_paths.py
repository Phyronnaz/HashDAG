import datetime
import os

root = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../")

def get_result_path(name):
    profiling_path = root + "/benchmark_results/" + name + "/" + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + "/"

    if not os.path.exists(profiling_path):
        os.makedirs(profiling_path)

    return profiling_path
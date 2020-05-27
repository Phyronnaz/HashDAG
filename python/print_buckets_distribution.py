import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog

from tools import results_prompt

root = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../")
initialdir = root + "/benchmark_results/bucket_count/"
path = filedialog.askdirectory(initialdir=initialdir) + "/"
print(path)

paths = []
for file in os.listdir(path):
    name, ext = os.path.splitext(os.path.splitext(os.path.basename(file))[0])
    if ext == ".before":
        print(file)
        paths.append(path + name)


def load_buckets(file):
    print("Loading " + file + "... ", end="")
    data = pd.read_csv(file, header=None, delimiter=";", dtype='unicode').values
    print("Loaded")

    level_buckets = {}
    level = -1
    for line in data:
        assert len(line) == 2, "line: {}".format(line)
        if line[0] == "level":
            level = int(line[1])
            level_buckets[level] = []
        else:
            level_buckets[level].append(int(line[1]))
    for level in list(level_buckets.keys()):
        if max(level_buckets[level]) == 0:
            del level_buckets[level]
        else:
            level_buckets[level] = np.array(level_buckets[level])

    return level_buckets


for path in paths:
    print("######################################")
    print("######################################")
    print("######################################")
    before_path = path + ".before.csv"
    after_path = path + ".after.csv"
    print(before_path, after_path)
    before = load_buckets(before_path)
    after = load_buckets(after_path)


    def print_buckets(buckets_map):
        for level, buckets in buckets_map.items():
            print("level {}: avg: {}; std: {}; min: {}; max: {}".format(
                level,
                np.mean(buckets),
                np.std(buckets),
                np.min(buckets),
                np.max(buckets)))


    print("BEFORE")
    print_buckets(before)
    print("AFTER")
    print_buckets(after)

import os
import numpy as np
import pandas as pd
from tkinter import filedialog

dags = []
names = []
defines = []
indices = None


def init(paths):
    for path in paths:
        name, ext = os.path.splitext(os.path.splitext(os.path.basename(path))[0])
        assert ext == ".stats", "Ext: " + ext

        print("Loading " + name + ".stats.csv... ", end="")
        # panda is WAY faster than numpy
        data = pd.read_csv(path, header=None, delimiter=",", dtype='unicode').values
        print("Loaded")

        print("Loading " + name + ".defines... ", end="")
        file_defines = {}
        with open(path[:-len(".stats.csv")] + ".defines", "r") as file:
            for line in file.readlines():
                if len(line) > 0 and line[-1] == '\n':
                    line = line[:-1]
                if len(line) == 0:
                    continue
                define, value = line.split("=", 1)
                # print("{}={}".format(define, value))
                file_defines[define] = value
        print("Loaded")

        names.append(name)
        dags.append(data)
        defines.append(file_defines)


def profiling_prompt():
    initialdir = os.path.realpath(os.path.realpath(__file__) + "/../../profiling/")
    paths = filedialog.askopenfiles(filetypes=[("CSV files", "*.stats.csv")],
                                    initialdir=initialdir)
    init([path.name for path in paths])


def results_prompt(name):
    root = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../")
    initialdir = root + "/benchmark_results/" + name
    path = filedialog.askdirectory(initialdir=initialdir) + "/"
    paths = []
    for file in os.listdir(path):
        if os.path.splitext(os.path.splitext(os.path.basename(file))[0])[1] == ".stats":
            print(file)
            paths.append(path + file)
    init(paths)
    return path


def get_arrays_median(datas, name):
    assert len(datas) > 0, "empty datas"
    global indices
    array = []
    for data in datas:
        filter = data[:, 1] == name
        array.append(data[filter][:, 2].astype(np.float))
        new_indices = data[filter][:, 0].astype(np.int)
        assert indices is None or np.all(indices == new_indices), \
            "Invalid stat identifier: " + name + ": {} num vs {}".format(len(new_indices), len(indices))
        indices = new_indices
    array = np.array(array)
    return np.median(array, axis=0)


def get_array(data, name):
    global indices
    filter = data[:, 1] == name
    new_indices = data[filter][:, 0].astype(np.int)
    assert indices is None or np.all(indices == new_indices), \
        "Invalid stat identifier: " + name + " {} num vs {}".format(len(new_indices), len(indices))
    indices = new_indices
    return data[filter][:, 2].astype(np.float)


def get_custom_array(data, name):
    global indices
    array = np.zeros(len(indices))
    for i in range(len(indices)):
        index = indices[i]
        d = data[data[:, 0].astype(np.int) == index]
        array[i] = float(d[d[:, 1] == name][0][2])
    return array

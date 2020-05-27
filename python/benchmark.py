import datetime
import os
import sys

from benchmark_paths import root
from benchmark_tools import run

args = {}

interesting_defines = []

for arg in sys.argv[1:]:
    name, value = arg.split("=")
    values = value.split(",")
    args[name] = values
    if len(values) > 1:
        interesting_defines.append(name)

print(args)

subfolder = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + "".join(["_" + define for define in interesting_defines])
subfolder_path = root + "/profiling/" + subfolder

os.mkdir(subfolder_path)

with open(subfolder_path + "/commandline.txt", "w") as file:
    file.write(str(args))

defines = [[]]
for name, values in args.items():
    new_defines = []
    for value in values:
        for define in defines:
            new_defines.append(define + [(name, value)])
    defines = new_defines

for define in defines:
    prefix = ""
    for name, value in define:
        if name in interesting_defines:
            if len(prefix) > 0:
                prefix += "_"
            prefix += "{}={}".format(name, value)
    run(define, prefix, subfolder_path)
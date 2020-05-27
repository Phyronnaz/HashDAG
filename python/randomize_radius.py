import os
from tkinter import filedialog
import random

initialdir = os.path.realpath(os.path.realpath(__file__) + "/../../replays/")
path = filedialog.askopenfile(filetypes=[("CSV files", "*.csv")],
                               initialdir=initialdir).name

print(path)

with open(path, "r") as src:
    with open(path + ".random", "w") as dest:
        lines = []
        for line in src.readlines():
            if line.startswith("EditSphere"):
                EditSphere, x, y, z, radius, bool = line.split(",")
                assert EditSphere == "EditSphere"
                radius = float(radius)
                radius = abs(radius + random.uniform(-4, 4))
                line = "{},{},{},{},{},{}".format(EditSphere, x, y, z, radius, bool)
            lines.append(line)
        dest.writelines(lines)

print("Done!")
# Hash DAG

This repo holds the source code of our paper 'Interactively Modifying Compressed Sparse Voxel Representations'.

The paper can be downloaded here: https://graphics.tudelft.nl/Publications-new/2020/CBE20/ModifyingCompressedVoxels-main.pdf

Video: https://youtu.be/GQAwDn1bh0E

Talk: https://youtu.be/ltkk_nlMhQo?t=254

## Abstract

Voxels are a popular choice to encode complex geometry. Their regularity makes updates easy and enables random retrieval of values. The main limitation lies in the poor scaling with respect to resolution. Sparse voxel DAGs (Directed Acyclic Graphs) overcome this hurdle and offer high-resolution representations for real-time rendering but only handle static data. We introduce a novel data structure to enable interactive modifications of such compressed voxel geometry without requiring de- and recompression. Besides binary data to encode geometry, it also supports compressed attributes (e.g., color). We illustrate the usefulness of our representation via an interactive large-scale voxel editor (supporting carving, filling, copying, and painting).

## Demo

A downloadable demo for Windows can be found here: https://drive.google.com/file/d/10vyCm39hC-Z-dnrEyBo2D4GIFlkPVOXc/view?usp=sharing

## Performance Note

The demo linked above uses the Epic Citadel at a resolution of 2^17. It will require a CUDA-capable GPU with at least 8GB of VRAM (6 might work too).

On Windows, the performance is degraded by the Windows Display Driver. Rendering should be at least twice as fast on Linux, for the same machine.

## Keys

```
Shift to go faster
R to reset replay
Shift R to clear replay
Backspace to save replay to disk
M to print allocated CUDA memory stats
Ctrl Z to undo, Ctrl Shift Z to redo
Tab to switch tools (Shift Tab goes the other way)
G to run garbage collection
U to clear undo history (free up memory)
Caps Lock to switch DAG
1/2/3/4/5/6/7/8/9/0: do debug stuff with colors
X to enable/disable shadows & fog
=/- to increase/decrease shadow bias
I/O to increase/decrease fog density
P to print stats
Shift P to print DAG/SVO stats (number of nodes etc)
Alpha pad: 0 to 9: go to predefined locations
Enter: print current location & rotation (can be copy pasted in engine.cpp to define new alpha key locations)
Keypad +: add 1 to the radius, useful to create edits benchmarks with different radius
H to hide UI
F1/F2/F3 to rotate
F4 to enable/disable swirl
F5 to increase (shift to decrease) swirl period
F6 to increase scale
```

## Creating DAGs

Compressed DAGs with colors can be created from meshes using the tool from Dan Dolonius: https://github.com/gegoggigog/DAG-example/tree/compression

Some additional work has been done in this fork: https://github.com/Phyronnaz/DAG_Compression

## Building from source

The code can be build on Windows using the included Visual Studio solution, and on Linux using cmake. You will need to install the latest CUDA release.

On Linux you'll need to install GLFW3 and GLEW.

You will need to download the binary files from here, and put them under `data`: https://drive.google.com/drive/folders/1sIYzKSAmOoMA9sfqzkpkF_LiN2HYKxxp?usp=sharing

## Settings

Most parameters are in `typedefs.h`. It is recommended to override them in `script_definitions.h`.

## Profiling

The code is hooked with the Tracy profiler. You can start it using `third_party\tracy\profiler\build\win32\Tracy.sln`.

Tracy docs can be found here: https://github.com/wolfpld/tracy

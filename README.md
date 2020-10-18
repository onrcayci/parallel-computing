# Parallel Computing #

Parallel Computing Practice using CUDA C/C++ libraries.

## Rectification ##

This CUDA file (rectify.cu) has been created using the Jupyter Notebook "rectification.ipynb" on Google Colab. However, it is possible to directly work on the rectify.cu file from a text editor.

## Logic Gates Simulation ##

This folder includes three different implementation of a Logic Gate Simulation program.

1. `sequential.c`: This is the sequential implementation of the logic gate simulation program. To use it, simply compile the program by running the command `gcc sequential.c -o build/sequential` and then calling the command `./build/sequential <input-textfile-name> <input-file-length> <output-textfile-name>`.

2. `parallelal_explicit.cu`: This is the parallel implementation of the logic gate simulation program using CUDA. Difference is that, this program uses explicit device memory allocation and memory transfer between the host (CPU) and the device (GPU). To use it, you need to make sure you have CUDA development kit installed. Then, you can compile the program by running the command `nvcc parallelal_explicit.cu -o build/parallelal_explicit`. You can then call run the program by running the command `./build/parallel_explicit <input-textfile-name> <input-file-length> <output-textfile-name>`.

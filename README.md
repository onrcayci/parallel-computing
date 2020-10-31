# Parallel Computing #

Parallel Computing Practice using CUDA C/C++ libraries.

## Rectification ##

This CUDA file (rectify.cu) has been created using the Jupyter Notebook "rectification.ipynb" on Google Colab. However, it is possible to directly work on the rectify.cu file from a text editor.

## Logic Gates Simulation ##

This folder includes four different implementations of a Logic Gate Simulation program.

1. `sequential.c`: This is the sequential implementation of the logic gate simulation program. To use it, simply compile the program by running the command `gcc sequential.c -o build/sequential` and then calling the command `./build/sequential <input-textfile-name> <input-file-length> <output-textfile-name>`.

2. `parallelal_explicit.cu`: This is the parallel implementation of the logic gate simulation program using CUDA. Difference is that, this program uses explicit device memory allocation and memory transfer between the host (CPU) and the device (GPU). To use it, you need to make sure you have CUDA development kit installed. Then, you can compile the program by running the command `nvcc parallelal_explicit.cu -o build/parallelal_explicit`. You can then call run the program by running the command `./build/parallel_explicit <input-textfile-name> <input-file-length> <output-textfile-name>`.

3. `parallelal_unified.cu`: This is another parallel implementation of the logic gate simulation program using CUDA. The difference between this and `parallelal_explicit.cu` is that, the unified memory is used in this implementation instead of the device memory. This means that we are allocating the arrays using `cudaMallocManaged` command. The array created by this command can be accessed by both the host and device. Thus, this makes things easier for the developers by only having one array that can be accessed by both of the systems.

4. `parallelal_prefetch.cu`: This parallel implementation is almost identical to `parallelal_unified.cu`, but there is one difference: the variables are prefetched to the device using the command `cudaMemPrefetchAsync`. This way, the performance of the parallel program is improved and the calculations are done in shorter times compared to the `parallelal_unified.cu`.

## Finite Element Music Synthesis ##

This folder includes three different implementations of a Finite Element Music Synthesis program. The main idea of the program is that, it is simulating a hit on a drum and calculates the position of each finite element in a 2D array. The program can be run for a specified number of iterations and changes in the 2D position array can be observed after each iteration.

1. `sequential.c`: This is the sequential implementation of the finite element music synthesis program. The size of 2D array used in this simulation is 4x4. This implementation of the program is used mainly to observe how this program can be parallelized using CUDA.

2. `grid_4_4.cu`: This is the parallel implementation of the finite element music synthesis program from the sequential part. Same 4x4 array is used again here. The program is parallelized by using  a 2D block grid. Each block has one thread and each thread calculate the position of the element that corresponds to their location in the block grid, i.e. the thread in block (0, 0) calculates the position of the finite element located at (0, 0).

3. `grid_512_512.cu`: This is bigger parallel implementation of the finite element music synthesis program. Instead of using the same 4x4 array, a 512x512 array is being used for simulation. Also, the parallelization scheme is also updated. Since not every CUDA GPU architecture is able to generate as many block or threads as this simulation needs, each thread is now responsible of a certain amoun of finite element calculation. The version on GitHub has a GPU grid structure of 4x4 block grid with 32x32 threads per block. Thus, each thread is responsible of 16 finite elements in the format of 4x4.

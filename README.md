# Parallel Computing

Parallel Computing Practice using CUDA C/C++ libraries.

## Rectification

This CUDA file (rectify.cu) has been created using the Jupyter Notebook "rectification.ipynb" on Google Colab. However, it is possible to directly work on the rectify.cu file from a text editor.

### Important Notes

- In order to use .c files in your CUDA code, you have to name change the extension of the file to ".cpp". Otherwise, the NVIDIA CUDA compiler `nvcc` cannot indetify the methods used from that file.

- `cudaMemcpy` is also tricky. I was passing the physical address of my pointer and this was causing an error in the code. However, you won't get an error message when this happens. The code compiles correctly and can be run on the terminal. I noticed this because I wasn't getting the result I was expecting to at the end, which was a png file as an output. Keep in mind while working with `cudaMemcpy` that sometimes passing a pointer or the physical address of the pointer won't work.

### Useful Links
https://developer.nvidia.com/blog/even-easier-introduction-cuda/

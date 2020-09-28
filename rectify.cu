// c libraries
#include <stdio.h>
#include <stdlib.h>
// cuda libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// image processing library
#include "lodepng.h"

// Device Code
__global__ void rectification(unsigned char *input_image, unsigned char *output_image, int width, int height, int size) {
    // thread's x coordinate in the block, corresponds to width
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    // thread's y coordinate in the block, corresponds to height
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < width && y < height) {
        // calculate the index of the pixel in the input image array
        int pixel_index = 4*width*y + 4*x;
        for (int i = 0; i < 4; i++) {
            if (pixel_index + i < size) {
                int value = (int) input_image[pixel_index+i];
                if (value < 127) value = 127;
                output_image[pixel_index+i] = (unsigned char) value;
            }
        }
    }
}

// Host Code
int main(int argc, char *argv[]) {
    
    if (argc <= 1) {
        return printf("No arguments provided! Please add input file name, output file name and thread number to the program call!");
    } else if (argc > 1 && argc < 4) {
        return printf("Missing arguments! Please check that you have provided the input file name, output file name and the number of threads!");
    }

    // get inputs from the command line
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    int threads_no = atoi(argv[3]);

    // initalize variables for error, input image, input image width and input image height
    unsigned error;
    unsigned char *input_image;
    unsigned width, height;

    // load input image from file to buffer array
    error = lodepng_decode32_file(&input_image, &width, &height, input_filename);

    printf("%d\n", error);
    
    // if there is an error while loading the file, return the error
    if(error) return printf("Error: %u: %s\n", error, lodepng_error_text(error));

    // initalize device variable to copy the input image over to the GPU
    unsigned char *d_input, *d_output;
    int size = width * height * 4 * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // copy image from host memory to device memory
    cudaMemcpy(d_input, input_image, size, cudaMemcpyHostToDevice);

    // initialize block size and block number to process the images
    dim3 blockSize(threads_no, threads_no);
    dim3 numBlocks(width/threads_no, height/threads_no);

    // run device kernel
    rectification<<<numBlocks, blockSize>>>(d_input, d_output, width, height, size);
    
    // synchronize device to get the output back from the device
    cudaDeviceSynchronize();

    // initialize output image array to copy output from device to host
    unsigned char *output_image = (unsigned char*)malloc(size);

    // copy output image from device to host
    cudaMemcpy(output_image, d_output, size, cudaMemcpyDeviceToHost);

    // save output image
    lodepng_encode32_file(output_filename, output_image, width, height);

    // free up device memory;
    cudaFree(d_input);
    cudaFree(d_output);
}
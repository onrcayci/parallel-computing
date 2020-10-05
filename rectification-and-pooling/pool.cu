
  #include "cuda_runtime.h"
  #include "device_launch_parameters.h"

  #include <stdio.h>
  #include <time.h>
  #include "lodepng.h"
  #define MAX_THREAD 1024

  __global__ void max_pooling(unsigned char* original_img, unsigned char* new_img, unsigned int width, unsigned int threads_no, unsigned int size) {
    unsigned int position;
    unsigned char max;
    
    // TODO: Fix logic bug
    for (int i = threadIdx.x; i < size/4; i = i + threads_no) {
      /*if (i%2==0){
        position = 4*i;
      }else{
          position = (4*i)-2;
      }*/

      position = i + (4 * (i / 4)) + (width * 4 * (i / (width * 2)));
      max = original_img[position];
      if (original_img[position + 4] > max)
        max = original_img[position + 4];
      if (original_img[position + width] > max)
        max = original_img[position + width];
      if (original_img[position + width + 4] > max)
        max = original_img[position + width + 4];

      new_img[i] = max;
    }
  }

  int main(int argc, char* argv[]) {
    
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
    unsigned char* input_img;
    unsigned width, height;
    
    // load input image from file to buffer array
    error = lodepng_decode32_file(&input_img, &width, &height, input_filename);

    // if there is an error while loading the file, return the error
    if (error) {
      return printf("Error: %u: %s\n", error, lodepng_error_text(error));
    }

    // initalize device variable to copy the input image over to the GPU
    unsigned char *d_input, *d_output;
    unsigned int input_size = width * height * 4 * sizeof(unsigned char);
    unsigned int output_size = input_size/4;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, output_size);
    
    // create CUDA events to time the kernel runtime
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // copy image from host memory to device memory
    cudaMemcpy(d_input, input_img, input_size, cudaMemcpyHostToDevice);

    // record start time
    cudaEventRecord(start);

    // run device kernel
    max_pooling<< <1, threads_no >> > (d_input, d_output, width, threads_no, input_size);

    // record stop time
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    // initialize output image array to copy output from device to host
    unsigned char* output_img = (unsigned char*)malloc(output_size);

    // copy output image from device to host
    cudaMemcpy(output_img, d_output, output_size, cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // save output image
    error = lodepng_encode32_file(output_filename, output_img, width/2, height/2);
    
    // if there is an error while loading the file, return the error
    if (error) {
      return printf("Error: %u: %s\n", error, lodepng_error_text(error));
    }

    //print elapsed time
    printf("Time Elapsed: %f ms\n", milliseconds);

    //free up host memory
    free(input_img);
    free(output_img);

    //free up device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
  }
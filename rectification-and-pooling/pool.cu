// C libraries
#include <stdio.h>
#include <stdlib.h>
// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// image processing library
#include "lodepng.h"


/* GPU function */
__global__ void pool(unsigned char* new_img, unsigned char* output_img, int xNumSections, int yNumSections, int input_img_width, int input_img_height) {

  // thread index
  int thread_index = threadIdx.x;

  // thread section index
  int column = thread_index % xNumSections;
  int row = thread_index / xNumSections;
  
  // thread section_width and section_height 
  int sec_w_temp = (input_img_width * 4 / xNumSections);
  int sec_h_temp = (input_img_height / yNumSections);
  int section_width = sec_w_temp - (sec_w_temp%2);
  int section_height = sec_h_temp - (sec_h_temp%2);

  for (int i = 0; i <= section_height; i += 2) {
    
    // index of output_img array and offset
    int output_img_index = section_width * column / 2 + (i/2 + section_height / 2 * row) *input_img_width * 2;
    int offset = output_img_index % 4;
    output_img_index = output_img_index - offset;

    for (int j = 0 - offset; j <= section_width + 4 - offset; j += 8) {
      
      // Calculate global pixel location indexes for new_img[]
      int global_row = i + section_height * row;
      int global_column = j + section_width * column;
      
      global_column = global_column - global_column % 4;
      
      // there are 4 values for a pixel: R, G, B, A. Loop over all of them to pool pixel      
      for (int rgba_val = 0; rgba_val < 4; rgba_val++) {
    
        // retrieve indexes of 2x2 square
        int upper_left_index = global_column + rgba_val + (global_row * input_img_width * 4);
        int upper_right_index = upper_left_index +4;
        int lower_left_index = global_column + rgba_val + (global_row + 1) * (input_img_width *4);
        int lower_right_index = lower_left_index + 4;
        
        // variables to stores values of 2x2 square 
        int upper_left_value = 0;
        int upper_right_value = 0; 
        int lower_left_value = 0; 
        int lower_right_value = 0;
     
        // Get value of each corner of 2x2 region, checking for edges
        if (upper_left_value < input_img_width * input_img_height * 4) {
          upper_left_value = new_img[upper_left_index];
        }
        if (global_column + rgba_val + 4 < input_img_width * 4) {
          upper_right_value = new_img[upper_right_index];
        }
        if (lower_left_index < input_img_width * input_img_height * 4) {
          lower_left_value = new_img[lower_left_index];
        }
        
        if (global_column + rgba_val + 4 < input_img_width * 4 && lower_right_index <input_img_width * input_img_height * 4) {
          lower_right_value = new_img[lower_right_index];
        }
        
        // Finding maximum value for the square
        int maximum_value = (upper_left_value >= upper_right_value? upper_left_value : upper_right_value);
        maximum_value = (maximum_value >= lower_left_value? maximum_value : lower_left_value);
        maximum_value = (maximum_value >= lower_right_value? maximum_value : lower_right_value);
      
        int input_img_area = input_img_height * input_img_width;
        
        if (output_img_index >= input_img_area) {
          break;
        }
        
        output_img[output_img_index++] = maximum_value;
        
      }       
    }
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
  unsigned char* new_img, *output_img;
  unsigned int input_size = width * height * 4 * sizeof(unsigned char);
  unsigned int output_size = input_size/4;

  // Get number of sections using number of threads
  int threads_no_copy = threads_no;
  int xNumSections = 1;
  int yNumSections = 1;
  bool side = true;

  while (threads_no_copy > 1) {
    if (!side) {
      yNumSections= yNumSections * 2;
    } else {
      xNumSections= xNumSections * 2;
    }
    side = !side;
    threads_no_copy = threads_no_copy / 2;
  }

  // record start time
  clock_t start = clock();

  // Allocating space for images
  cudaMallocManaged((void**)& new_img, input_size * sizeof(unsigned char));
  cudaMallocManaged((void**)& output_img, output_size * sizeof(unsigned char));
  
  // Setting up data array for GPU
  for (int i = 0; i < input_size; i++) {
    new_img[i] = input_img[i];
  }

  // Launch pool() kernel on GPU with threads_no threads
  pool << <1, threads_no >> > (new_img, output_img, xNumSections, yNumSections, width, height);
  
  // Wait for GPU threads to complete
  cudaDeviceSynchronize();

  // print time passed
  printf("%ld msec", clock() - start);

  // save output image
  lodepng_encode32_file(output_filename, output_img, width/2, height/2);
  
  // free up memory
  free(input_img);
  cudaFree(new_img);
  cudaFree(output_img);

  return 0;
}
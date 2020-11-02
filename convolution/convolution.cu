
// C libraries
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// Image Processing Library
#include "lodepng.h"
// Weight Matrix
#include "wm.h"

__global__ void convolution(unsigned char* img, unsigned char* convoluted_img, int block_count, int thread_count, int width, int height, float weight_matrix[3][3])
{

	int index = threadIdx.x + blockIdx.x * thread_count;
	int weight_matrix_size = 3;

	// declare variables for number of rows per thread, start row, end row, start col, and end col
	float num_rows_per_thread, start_row, end_row;
	int start_col, end_col;

	int total_thread_no = block_count * thread_count;

	// Less/equal number of threads to height => only make row sections
	if (total_thread_no<= height){
		num_rows_per_thread =	(float)height / (float)(block_count * thread_count);
		start_row = index * num_rows_per_thread;
		end_row = (index+1) * num_rows_per_thread;

		// fix starting row for index = 0 since range for rows is 1 to m-1
		if (index == 0) {
			start_row = 1;
		}
		// fix ending row for last index since range for rows is 1 to m-1
		if (index == block_count * thread_count - 1) {
			end_row = end_row - 1;
		}

		// set starting column to be 1 since range for columns is 1 to n-1
		start_col = 1;
		// set ending column to be n-1 since range for columns is 1 to n-1
		end_col = width - 1;
	}
	// More number of threads than number of rows => make row and column sections
	else  {
		// determine number of column sections
		int num_cols = ((total_thread_no) / height) + 1;

		// Adjust starting row and ending row
		num_rows_per_thread = ((float)height / (float)(block_count * thread_count)) * num_cols;
		start_row = (index/num_cols) * num_rows_per_thread;
		end_row = (index / num_cols + 1) * num_rows_per_thread;
		
		// fix start row edge case
		if (start_row == 0) {
			start_row = 1;
			end_row = start_row + num_rows_per_thread;
		}
		// fix end row edge case
		if (end_row > height - weight_matrix_size / 2) {
			end_row = end_row - weight_matrix_size / 2;
		}
		
		// Adjust starting column and ending column
		start_col = (index % num_cols) * width / num_cols;
		end_col = (index % num_cols + 1) * width / num_cols;
		
		// fix starting column for edge case
		if (index % num_cols == num_cols-1) {
			start_col = (index % num_cols) * width / num_cols;
		}
		// fix ending column for edge case
		if (index % num_cols == 0) {
			end_col = width / num_cols; 
		}
	}

	// typecast the row values from float to int to use in the loop
  int start_row_int = (int) start_row;
	int end_row_int = (int) end_row;

	// iterate over the image under the given bounds specificed by starting row, ending row, starting col, and ending col
	for (int i = start_row_int; i < end_row_int; i++) {
		for (int j = start_col; j < end_col; j++) {
			
			// There are 4 values for a pixel: R, G, B, A. Loop over all of them to convolute pixels     
			for (int rgba_val = 0; rgba_val < 4; rgba_val++) {

				// To store the output value
				float output_total = 0;
				
				// alpha channel
				if (rgba_val == 3){
						output_total = img[i * width * 4 + (j * 4) + rgba_val];
				}
				// rgb channels
				else {
					// use the weight matrix to multiply	
					for (int ii = 0; ii < weight_matrix_size; ii++) {
						for (int jj = 0; jj < weight_matrix_size; jj++) {
							// use formula given in the lab assignment
							output_total = output_total + img[(i + ii - 1) * width * 4 + (j * 4 + (jj - weight_matrix_size/2) * 4) + rgba_val] * weight_matrix[ii][jj];
						}
					}
					// clamping values 
					if (output_total < 0) {
						output_total = 0;
					} else if (output_total > 255) {
						output_total = 255;
					}
				}

				// store result in convoluted image
				int convoluted_img_index = (i - 1) * (width - (weight_matrix_size - 1)) * 4 + (j - weight_matrix_size / 2)* 4 + rgba_val;
				convoluted_img[convoluted_img_index] = (unsigned char)round(output_total);
			}
		}
	}
}

int main(int argc, char* argv[])
{
		
	// check if the arguments are valid
	if (argc <= 1) {
      return printf("No arguments provided! Please add input file name, output file name and thread number to the program call!");
  } else if (argc > 1 && argc < 4) {
      return printf("Missing arguments! Please check that you have provided the input file name, output file name and the number of threads!");
  }

  // get inputs from the command line
  char *input_filename = argv[1];
  char *output_filename = argv[2];
  int thread_count = atoi(argv[3]);

	// declare variables for error, input image, convoluted image, input image width, input image height, input image length, and convoluted image length
	unsigned error;
	unsigned char* input_img, * new_img, * convoluted_img;
	unsigned width, height;
	int img_length, convoluted_img_length;

	// declare weight matrix and initalize its size
	float* weight_matrix;
	int weight_matrix_size = 3;

	// load input image from file to buffer array
	error = lodepng_decode32_file(&input_img, &width, &height, input_filename);

	// if there is an error while loading the file, return the error
	if (error) {
		return printf("Error: %u: %s\n", error, lodepng_error_text(error));
	}

	// Calculate length of the loaded image
	img_length = width * height * 4 * sizeof(unsigned char);
	// Calculate length of convoluted image
	convoluted_img_length = (width - (weight_matrix_size - 1)) * (height - (weight_matrix_size - 1)) * 4;

	// Allocating space for input image, convoluted image, and the weight matrix
	cudaMallocManaged((void**)& new_img, img_length * sizeof(unsigned char));
	cudaMallocManaged((void**)& convoluted_img, convoluted_img_length * sizeof(unsigned char));
	cudaMallocManaged((void**)& weight_matrix, weight_matrix_size * weight_matrix_size * sizeof(float));

	// Initialize image data array for GPU 
	for (int i = 0; i < img_length; i++) {
		new_img[i] = input_img[i];
	}

	int block_count = 1;
	// Fix block count based on the threads given
	if (thread_count > 1024) {
		block_count = (thread_count / 1024) + 1;
		thread_count = thread_count / block_count;
	}
	
	int weight_matrix_num_elements = weight_matrix_size * weight_matrix_size;
	std::copy(&w[0][0], &w[0][0] + weight_matrix_num_elements, weight_matrix);

	// Launch convolution() kernel on GPU with thread_count threads
	convolution << <block_count, thread_count >> > (new_img, convoluted_img, block_count, thread_count, width, height, reinterpret_cast<float(*)[3]>(weight_matrix));

	// Wait for GPU threads to complete
	cudaDeviceSynchronize();

	// Save output image
	lodepng_encode32_file(output_filename, convoluted_img, width - 2, height - 2);

	// Free up memory
	free(input_img);
	cudaFree(new_img);
	cudaFree(convoluted_img);
	cudaFree(weight_matrix);

  return 0;
}
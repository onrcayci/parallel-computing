#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

__global__ void computeLogicGates(char* d_input, char* d_output, int size) {
    // calculate the index of the thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int input_index = index * 3;
    // if the index is inside the range of the array
    if (input_index < size) {
        int output;
        switch (d_input[input_index+2] - '0') {
            case AND:
                if (d_input[input_index] == '1' && d_input[input_index+1] == '1') output = 1;
                else output = 0;
                break;
            case OR:
                if (d_input[input_index] == '0' && d_input[input_index+1] == '0') output = 0;
                else output = 1;
                break;
            case NAND:
                if (d_input[input_index] == '1' && d_input[input_index+1] == '1') output = 0;
                else output = 1;                
                break;
            case NOR:
                if (d_input[input_index] == '0' && d_input[input_index+1] == '0') output = 1;
                else output = 0;                
                break;
            case XOR:
                if (d_input[input_index] == d_input[input_index+1]) output = 0;
                else output = 1;
                break;
            case XNOR:
                if (d_input[input_index] == d_input[input_index+1]) output = 1;
                else output = 0;                
                break;                
        }
        d_output[index] = output + '0';
    }
}

int main(int argc, char* argv[]) {
    // check if necessary arguments are provided
    if (argc == 1) {
        return printf("No arguments are provided! Please provide the input file path, input file length and the output file path!");
    }
    else if (argc == 2) {
        return printf("Input file length and output file path are not provided!");
    }
    else if (argc == 3) {
        return printf("Output file path is not provided!");
    }

    char* input_file = argv[1];
    int input_size = atoi(argv[2]);
    char* output_file = argv[3];

    // read the input file
    FILE* input_fptr;
    input_fptr = fopen(input_file, "r");
    if (!input_fptr) return printf("Error opening the input file!");

    // read the file line by line and populate input_data array
    char line[10];

    // allocate CUDA variables
    char* d_input;
    char* d_output;
    int input_array_size = input_size * 3 * sizeof(char);
    int output_array_size = input_size * sizeof(char);

    cudaMallocManaged(&d_input, input_array_size);
    cudaMallocManaged(&d_output, output_array_size);

    for (int i = 0; i < input_size; i++) {
        fgets(line, 9, input_fptr);
        d_input[i*3] = line[0];
        d_input[i*3+1] = line[2];
        d_input[i*3+2] = line[4];
    }

    // prefetch the data to the GPU
    int device = -1;
    cudaGetDevice(&device);

    clock_t start = clock(); 

    cudaMemPrefetchAsync(d_input, input_array_size, device, NULL);
    cudaMemPrefetchAsync(d_output, output_array_size, device, NULL);

    // call device kernel
    computeLogicGates<<<input_size, 1>>>(d_input, d_output, input_array_size);

    // synchronize threads
    cudaDeviceSynchronize();

    clock_t end = clock();

    // write the results into the output file
    FILE* output_fptr;
    output_fptr = fopen(output_file, "w");
    if(!output_fptr) return printf("Error opening output file!");
    for (int i = 0; i < input_size; i++) {
        char data[3];
        sprintf(data, "%c\n", d_output[i]);
        fputs(data, output_fptr);
    }

    // close file pointer
    fclose(output_fptr);

    // free up device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // calculate execution time
    double runtime = (double) (end-start) / CLOCKS_PER_SEC;
    printf("Execution time: %f ms\n", runtime * 1000);

    return 0;
}
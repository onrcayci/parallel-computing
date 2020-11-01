#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// declare device constants
__constant__ float n = 0.0002;
__constant__ float p = 0.5;
__constant__ float G = 0.75;
// 2D finite element grid for one thread (ept x ept)
__constant__ int ept = 8;

// declare global constants
const int N = 512;
const dim3 blocks = dim3(8, 8);
const dim3 threads = dim3(8, 8);

// parallelize inner element position calculation
__global__ void calculate_inner_elements(float* d_u, float* d_u1, float* d_u2) {
    int t_i = threadIdx.x + blockIdx.x * blockDim.x;
    int t_j = threadIdx.y + blockIdx.y * blockDim.y;
   for (int h = 0; h < ept; h++) {
        for (int w = 0; w < ept; w++) {
            int i = ept * t_i + h;
            int j = ept * t_j + w;
            if (i >= 1 && i <= (N - 2) && j >= 1 && j <= (N - 2)) {
                d_u[N*i+j] = (p * (d_u1[N*(i-1)+j] + d_u1[N*(i+1)+j] + d_u1[N*i+(j-1)] + d_u1[N*i+(j+1)] - (4 * d_u1[N*i+j]))
                                + (2 * d_u1[N*i+j]) - ((1 - n) * d_u2[N*i+j])) / (1 + n);
            }
        }
    }
}

// parallelize edge element position calculation
__global__ void calculate_edge_elements(float* d_u) {
    int t_i = threadIdx.x + blockIdx.x * blockDim.x;
    int t_j = threadIdx.y + blockIdx.y * blockDim.y;
    for (int h = 0; h < ept; h++) {
        for (int w = 0; w < ept; w++) {
            int i = ept * t_i + h;
            int j = ept * t_j + w;
            // top edge
            if (i == 0 && j >= 1 && j <= N - 2) {
                d_u[N*i+j] = G * d_u[N*(i+1)+j];
            }
            else if (i >= 1 && i <= N - 2) {
                // left edge
                if (j == 0) {
                    d_u[N*i+j] = G * d_u[N*i+1];
                }
                // right edge
                else if (j == N - 1) {
                    d_u[N*i+j] = G * d_u[N*i+(j-1)];
                }
            }
            // bottom edge
            else if (i == N - 1 && j >= 1 && j <= N - 2) {
                d_u[N*i+j] = G * d_u[N*(i-1)+j];
            }
        }
    }
}

// parallelize corner element position calculation
__global__ void calculate_corner_elements(float* d_u) {
    int t_i = threadIdx.x + blockIdx.x * blockDim.x;
    int t_j = threadIdx.y + blockIdx.y * blockDim.y;
    for (int h = 0; h < ept; h++) {
        for (int w = 0; w < ept; w++) {
            int i = ept * t_i + h;
            int j = ept * t_j + w;
            if (i == 0) {
                // top left corner
                if (j == 0) {
                    d_u[N*i+j] = G * d_u[N*(i+1)+j];
                }
                // top right corner
                else if (j == N - 1) {
                    d_u[N*i+j] = G * d_u[N*i+(j-1)];
                }
            }
            else if (i == N - 1) {
                // bottom left corner
                if (j == 0) {
                    d_u[N*i+j] = G * d_u[N*(i-1)+j];
                }
                // bottom right corner
                else if (j == N - 1) {
                    d_u[N*i+j] = G * d_u[N*i+(j-1)];
                }
            } 
        }
    }
}

// parallelize positional array updates u2 = u1 and u1 = u
__global__ void update_positional_arrays(float* d_u, float* d_u1, float* d_u2) {
    int t_i = threadIdx.x + blockIdx.x * blockDim.x;
    int t_j = threadIdx.y + blockIdx.y * blockDim.y;
    for (int h = 0; h < ept; h++) {
        for (int w = 0; w < ept; w++) {
            int i = ept * t_i + h;
            int j = ept * t_j + w;
            if(i < N && j < N) {
                int index = N * i + j;
                d_u2[index] = d_u1[index];
                d_u1[index] = d_u[index];
            }
        }
    }
}

int main(int argc, char* argv[]) {

    // check if the iteration value T is provided
    if (argc < 2) {
        return printf("The iteration value T is not provided!\n");
    }

    // get the iteration value T from the command line
    int iteration = atoi(argv[1]);

    // instantiate the 2D u, u1 and u2 matrices
    float* u = (float*) malloc(N * N * sizeof(float));
    float* u1 = (float*) malloc(N * N * sizeof(float));
    float* u2 = (float*) malloc(N * N * sizeof(float));

    // populate the matrices with zeros
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int index = N * i + j;
            u[index] = 0;
            if (i == N / 2 && j == N / 2) u1[index] = 1;
            else u1[index] = 0;
            u2[index] = 0;
        }
    }

    // instantiate device variables
    float* d_u;
    float* d_u1;
    float* d_u2;

    // allocate device memory for the variables
    cudaMalloc(&d_u, N * N * sizeof(float));
    cudaMalloc(&d_u1, N * N * sizeof(float));
    cudaMalloc(&d_u2, N * N * sizeof(float));

    // print out the size of the grid
    printf("Size of grid: %d nodes\n", N*N);

    clock_t start = clock();

    // start simulation
    for (int T = 0; T < iteration; T++) {

        // copy the variables from host to device
        cudaMemcpy(d_u, u, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u1, u1, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u2, u2, N * N * sizeof(float), cudaMemcpyHostToDevice);
        
        // call the device kernel function
        calculate_inner_elements<<<blocks, threads>>>(d_u, d_u1, d_u2);
        calculate_edge_elements<<<blocks, threads>>>(d_u);
        calculate_corner_elements<<<blocks, threads>>>(d_u);
        update_positional_arrays<<<blocks, threads>>>(d_u, d_u1, d_u2);

        cudaDeviceSynchronize();

        // copy the results back to host
        cudaMemcpy(u, d_u, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(u1, d_u1, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(u2, d_u2, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        // print u[N/2][N/2]
        printf("(%d, %d): %f\n", N/2, N/2, u[N*(N/2)+(N/2)]);

    } // end simulation

    clock_t end = clock();

    // calculate runtime and print out the result
    double runtime = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f ms\n", runtime * 1000);

    // free up the host memory used by the positional arrays
    free(u);
    free(u1);
    free(u2);

    // free up the device memory used by the positional arrays
    cudaFree(d_u);
    cudaFree(d_u1);
    cudaFree(d_u2);

    return 0;
}
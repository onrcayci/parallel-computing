#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// declare device constants
__constant__ float n = 0.0002;
__constant__ float p = 0.5;
__constant__ float G = 0.75;

// declare global constants
const int N = 4;

// parallelize inner element position calculation
__global__ void calculate_inner_elements(float* d_u, float* d_u1, float* d_u2) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= 1 && i <= N - 2 && j >= 1 && j <= N - 2) {
        // calculate the positions of the inner elements
        d_u[N*i+j] = (p * (d_u1[N*(i-1)+j] + d_u1[N*(i+1)+j] + d_u1[N*i+(j-1)] + d_u1[N*i+(j+1)] - (4 * d_u1[N*i+j]))
                        + (2 * d_u1[N*i+j]) - ((1 - n) * d_u2[N*i+j])) / (1 + n);
    } 
}

// parallelize edge element position calculation
__global__ void calculate_edge_elements(float* d_u) {
    int i = blockIdx.x;
    int j = blockIdx.y;
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

// parallelize corner element position calculation
__global__ void calculate_corner_elements(float* d_u) {
    int i = blockIdx.x;
    int j = blockIdx.y;
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

// parallelize positional array updates u2 = u1 and u1 = u
__global__ void update_positional_arrays(float* d_u, float* d_u1, float* d_u2) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if(i < N && j < N) {
        int index = N * i + j;
        d_u2[index] = d_u1[index];
        d_u1[index] = d_u[index];
    }
}

// parallelize positional array initialization
__global__ void create_positional_arrays(float* d_u, float* d_u1, float* d_u2) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if(i < N && j < N) {
        int index = N * i + j;
        d_u[index] = 0;
        if (i == N/2 && j == N/2) d_u1[index] = 1;
        else d_u1[index] = 0;
        d_u2[index] = 0;
    }
}

int main(int argc, char* argv[]) {

    // check if the iteration value T is provided
    if (argc < 2) {
        return printf("The iteration value T is not provided!\n");
    }

    // get the iteration value T from the command line
    int iteration = atoi(argv[1]);

    // instantiate the 2D u matrix
    float* u = (float*) malloc(N * N * sizeof(float));

    // print out the size of the grid
    printf("Size of grid: %d nodes\n", N*N);

    // instantiate device variables
    float* d_u;
    float* d_u1;
    float* d_u2;

    // allocate device memory for the variables
    cudaMalloc(&d_u, N * N * sizeof(float));
    cudaMalloc(&d_u1, N * N * sizeof(float));
    cudaMalloc(&d_u2, N * N * sizeof(float));

    // 2D block structure
    dim3 blocks = dim3(N, N);

    create_positional_arrays<<<blocks, 1>>>(d_u, d_u1, d_u2);

    cudaDeviceSynchronize();

    // start simulation
    for (int T = 0; T < iteration; T++) {
        
        // call the device kernel function
        calculate_inner_elements<<<blocks, 1>>>(d_u, d_u1, d_u2);
        calculate_edge_elements<<<blocks, 1>>>(d_u);
        calculate_corner_elements<<<blocks, 1>>>(d_u);
        update_positional_arrays<<<blocks, 1>>>(d_u, d_u1, d_u2);

        cudaDeviceSynchronize();

        // copy the results back to host
        cudaMemcpy(u, d_u, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        // print u[N/2][N/2]
        printf("(%d, %d): %f\n", N/2, N/2, u[N*(N/2)+(N/2)]);

    } // end simulation

    // free up the host memory used by the positional arrays
    free(u);

    // free up the device memory used by the positional arrays
    cudaFree(d_u);
    cudaFree(d_u1);
    cudaFree(d_u2);

    return 0;
}
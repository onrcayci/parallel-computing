#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// instantiate coefficients
const float n = 0.0002;
const float p = 0.5;
const float G = 0.75;
const int N = 4;

int main(int argc, char* argv[]) {

    // check if the iteration value T is provided
    if (argc < 2) {
        return printf("The iteration value T is not provided!\n");
    }

    // get the iteration value T from the command line
    int iteration = atoi(argv[1]);

    // instantiate the 2D u, u1 and u2 matrices
    float** u = (float**) malloc(N * sizeof(float*));
    float** u1 = (float**) malloc(N * sizeof(float*));
    float** u2 = (float**) malloc(N * sizeof(float*));

    // populate the matrices with zeros
    for (int i = 0; i < N; i++) {
        u[i] = (float*) malloc(N * sizeof(float));
        u1[i] = (float*) malloc(N * sizeof(float));
        u2[i] = (float*) malloc(N * sizeof(float));
        for (int j = 0; j < N; j++) {
            u[i][j] = 0;
            if (i == 2 && j == 2) u1[i][j] = 1;
            else u1[i][j] = 0;
            u2[i][j] = 0;
        }
    }

    printf("Size of grid: %d nodes\n", N*N);

    clock_t start = clock();

    // start simulation
    for (int T = 0; T < iteration; T++) {
        // calculate the position of the interior elements
        for (int i = 1; i <= N-2; i++) {
            for (int j = 1; j <= N-2; j++) {
                u[i][j] = (p * (u1[i-1][j] + u1[i+1][j] + u1[i][j-1] + u1[i][j+1] - (4 * u1[i][j])) + (2 * u1[i][j]) - ((1 - n) * u2[i][j])) / (1 + n);
            }
        }

        // ensure boundary conditions are met by the side elements
        for (int i = 1; i <= N-2; i++) {
            u[0][i] = G * u[1][i];
            u[N-1][i] = G * u[N-2][i];
            u[i][0] = G * u[i][1];
            u[i][N-1] = G * u[i][N-2];
        }

        // ensure boundary conditions are met by the corner elements
        u[0][0] = G * u[1][0];
        u[N-1][0] = G * u[N-2][0];
        u[0][N-1] = G * u[0][N-2];
        u[N-1][N-1] = G * u[N-1][N-2];

        // update the positional arrays
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                u2[i][j] = u1[i][j];
                u1[i][j] = u[i][j];
            }
        }

        // print out u(N/2, N/2)
        printf("(%d, %d): %f\n", N/2, N/2, u[N/2][N/2]);

    } // end simulation

    clock_t end = clock();
    
    double runtime = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f ms\n", runtime * 1000);

    // free up the memory used my the positional arrays
    free(u);
    free(u1);
    free(u2);

    return 0;
}
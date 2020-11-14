#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// method to read the first, second and fourth input files
int read_input_one_two_four(int** input, char* filepath) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) return fprintf(stderr, "Couldn't open file for reading\n");

    int file_length;
    fscanf(fp, "%d", &file_length);
    *input = (int*) malloc(file_length * sizeof(int));

    int next_int;

    for (int i = 0; i < file_length; i++) {
        fscanf(fp, "%d", &next_int);
        (*input)[i] = next_int;
    }

    fclose(fp);
    return file_length;
}

// method to read the third input file
int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) return fprintf(stderr, "Couldn't open file for reading\n");

    int file_length;
    fscanf(fp, "%d", &file_length);

    *input1 = (int*) malloc(file_length * sizeof(int));
    *input2 = (int*) malloc(file_length * sizeof(int));
    *input3 = (int*) malloc(file_length * sizeof(int));
    *input4 = (int*) malloc(file_length * sizeof(int));

    int next_int1;
    int next_int2;
    int next_int3;
    int next_int4;

    for (int i = 0; i < file_length; i++) {
        fscanf(fp, "%d, %d, %d, %d", &next_int1, &next_int2, &next_int3, &next_int4);
        (*input1)[i] = next_int1;
        (*input2)[i] = next_int2;
        (*input3)[i] = next_int3;
        (*input4)[i] = next_int4;
    }

    fclose(fp);
    return file_length;

}

int gate_solver(int node_gate, int node_output, int node_input) {
    switch (node_gate) {
        case 0:
            if (node_output == 1 && node_input == 1) return 1;
            return 0;
        case 1:
            if (node_output == 0 && node_input == 0) return 0;
            return 1;
        case 2:
            if (node_output == 1 && node_input == 1) return 0;
            return 1;
        case 3:
            if (node_output == 0 && node_input == 0) return 1;
            return 0;
        case 4:
            if (node_output == node_input) return 0;
            return 1;
        case 5:
            if (node_output == node_input) return 1;
            return 0;               
    }  
}

int main(int argc, char* argv[]) {  
    if (argc < 7) {
        return fprintf(stderr, "Missing input argument(s)!\n");
    }

    // get input file paths from command line
    char* input1 = argv[1];
    char* input2 = argv[2];
    char* input3 = argv[3];
    char* input4 = argv[4];
    char* output1 = argv[5];
    char* output2 = argv[6];

    // variables
    int num_node_ptrs;
    int num_nodes;
    int* node_ptrs;
    int* node_neighbors;
    int* node_visited;
    int num_total_teighbors;
    int* curr_level_nodes;
    int num_curr_level_nodes;
    int num_next_level_nodes = 0;
    int* node_gate;
    int* node_input;
    int* node_output;

    num_node_ptrs = read_input_one_two_four(&node_ptrs, input1);
    num_total_teighbors = read_input_one_two_four(&node_neighbors, input2);
    num_nodes = read_input_three(&node_visited, &node_gate, &node_input, &node_output, input3);
    num_curr_level_nodes = read_input_one_two_four(&curr_level_nodes, input4);

    // output
    int* next_level_nodes = (int*) malloc(num_nodes * sizeof(int));

    clock_t start = clock();

    // loop over all nodes in the current level
    for (int i = 0; i < num_curr_level_nodes; i++) {
        int node = curr_level_nodes[i];
        //loop over all neighbors of the node
        for (int j = node_ptrs[node]; j < node_ptrs[node+1]; j++) {
            int neighbor = node_neighbors[j];
            // if the neighbor hasn't been visited yet
            if (!node_visited[neighbor]) {
                // mark it and add it to the queue
                node_visited[neighbor] = 1;
                node_output[neighbor] = gate_solver(node_gate[neighbor], node_output[node], node_input[neighbor]);
                next_level_nodes[num_next_level_nodes] = neighbor;
                ++num_next_level_nodes;
            }
        }
    }

    clock_t end = clock();

    FILE* node_output_file = fopen(output1, "w");
    FILE* next_level_nodes_file = fopen(output2, "w");

    if(!node_output_file || !next_level_nodes_file) return fprintf(stderr, "Error opening the output files!");

    // write the length of the files
    fprintf(node_output_file, "%d\n", num_nodes);
    fprintf(next_level_nodes_file, "%d\n", num_next_level_nodes);

    for (int i = 0; i < num_nodes; i++) {
        fprintf(node_output_file, "%d\n", node_output[i]);
    }

    for (int i = 0; i < num_next_level_nodes; i++) {
        fprintf(next_level_nodes_file, "%d\n", next_level_nodes[i]);
    }

    double runtime = (double) (end - start) / CLOCKS_PER_SEC * 1000;
    printf("Execution time: %f ms\n", runtime);

    return 0;
}
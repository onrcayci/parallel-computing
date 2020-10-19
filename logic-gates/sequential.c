#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

char* and_gate(char input1, char input2) {
    if (input1 == '1' && input2 == '1') return "1\n";
    return "0\n";
}

char* or_gate(char input1, char input2) {
    if (input1 == '0' && input2 == '0') return "0\n";
    return "1\n";
}

char* nand_gate(char input1, char input2) {
    if (input1 == '1' && input2 == '1') return "0\n";
    return "1\n";
}

char* nor_gate(char input1, char input2) {
    if (input1 =='0' && input2 == '0') return "1\n";
    return "0\n";
}

char* xor_gate(char input1, char input2) {
    if (input1 == input2) return "0\n";
    return "1\n";
}

char* xnor_gate(char input1, char input2) {
    if (input1 == input2) return "1\n";
    return "0\n";
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

    // open the file
    FILE* input_fptr;
    FILE* output_fptr;
    char line[10];
    input_fptr = fopen(input_file, "r");
    output_fptr = fopen(output_file, "w");

    if (!input_fptr) return printf("Error reading the input file!");
    if (!output_fptr) return printf("Error openning the output file!");

    clock_t start = clock();

    // read each line
    for (int i = 0; i < input_size; i++) {
        // get the next line
        fgets(line, 9, input_fptr);
        // check the logic gate number
        char* output;
        switch (line[4] - '0') {
            case AND:
                output = and_gate(line[0], line[2]);
                break;
            case OR:
                output = or_gate(line[0], line[2]);
                break;
            case NAND:
                output = nand_gate(line[0], line[2]);
                break;
            case NOR:
                output = nor_gate(line[0], line[2]);
                break;
            case XOR:
                output = xor_gate(line[0], line[2]);
                break;
            case XNOR:
                output = xnor_gate(line[0], line[2]);
                break;                
        }
        fputs(output, output_fptr);
    }

    clock_t end = clock();

    // close the file pointers
    fclose(input_fptr);
    fclose(output_fptr);

    // calculate execution time in seconds
    double runtime = (double) (end - start) / CLOCKS_PER_SEC;
    // convert it into milliseconds
    runtime = runtime * 1000;
    printf("Execution time: %f ms\n", runtime);

    return 0;
}
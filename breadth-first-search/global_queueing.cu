#include <stdio.h>
#include <stdlib.h>

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

__device__ int globalQueue[7000000];
__device__ int numNextLevelNodes = 0;

__global__ void global_queuing_kernel(int totalThreads, int numNodes, int* nodePtrs, int* currLevelNodes, int* nodeNeighbors, int* nodeVisited, int* nodeGate, int* nodeInput, int* nodeOutput){
    
    int nodesPerThread = numNodes / totalThreads;
    int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
    int beginIndex = threadIndex * nodesPerThread;

    //Loop over all nodes in the current level
    for (int index = beginIndex; index < numNodes && index < beginIndex + nodesPerThread; index++) {
        
        int nodeIndex = currLevelNodes[index];

        //Loop over all neighbors of the node
        for (int secondIndex = nodePtrs[nodeIndex]; secondIndex < nodePtrs[nodeIndex+1]; secondIndex++) {   
            
            int neighborIndex = nodeNeighbors[secondIndex];
            const int alreadyVisited = atomicExch(&(nodeVisited[neighborIndex]),1);
            
            //If the neighbor hasn’t been visited yet
            if (!alreadyVisited) {
                
                int result = 0;
                int nInputV = nodeInput[neighborIndex];
                int nOutputV = nodeOutput[nodeIndex];
                int nGateV = nodeGate[neighborIndex];
                
                switch (nGateV) {
                case 0:
                  if (nInputV == 1 && nOutputV == 1) {
                      result = 1;
                  }
                  else {
                      result = 0;
                  }
                  break;
                case 1:
                  if (nInputV == 0 && nOutputV == 0) {
                      result = 0;
                  }
                  else {
                      result = 1;
                  }
                  break;
                case 2:
                  if (nInputV == 1 && nOutputV == 1) {
                      result = 0;
                  } else {
                      result = 1;
                  }
                  break;
                case 3:
                  if (nInputV == 0 && nOutputV == 0) {
                      result = 1; 
                  } else {
                      result = 0;
                  }
                  break;
                case 4:
                  if (nInputV == nOutputV) {
                      result = 0;
                  } else {
                      result = 1;
                  }
                  break;
                case 5:
                  if (nInputV == nOutputV) {
                      result = 1;
                  } else {
                      result = 0;
                  }
                  break;         
                }  

                //Update node output
                nodeOutput[neighborIndex] = result;
                int index = atomicAdd(&numNextLevelNodes,1); 
               
                //Add it to the global queue
                globalQueue[index] = neighborIndex; 
            }    
        }
         __syncthreads();
    }
}

inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
  }
  return err;
}

int main(int argc, char *argv[]){
    
    if (argc < 7) {
        return fprintf(stderr, "Missing input argument(s)!\n");
    }
    
    // Variables
    int numNodePtrs;
    int numNodes;
    int *nodePtrs_h;
    int *nodeNeighbors_h;
    int *nodeVisited_h;
    int numTotalNeighbors_h;
    int *currLevelNodes_h;
    int numCurrLevelNodes;
    int numNextLevelNodes_h;
    int *nodeGate_h;
    int *nodeInput_h;
    int *nodeOutput_h;
    
    numNodePtrs = read_input_one_two_four(&nodePtrs_h, argv[1]);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, argv[2]);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h,argv[3]);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, argv[4]);
    
    char* nodeOutput_fileName = argv[5];
    char* nextLevelNodes_fileName = argv[6];
    
    // output
    int *nextLevelNodes_h = (int *)malloc(numNodes*sizeof(int));
    
    checkCudaErr(cudaMemcpyToSymbol(globalQueue,nextLevelNodes_h, numNodes * sizeof(int)), "Copying");
    
    int numNodesSize = numNodes * sizeof(int);
    int numBlocks = 35;
    int blockSize = 128;
    
    // Cuda variables
    int* nodePtrs_cuda = (int*)malloc( numNodePtrs * sizeof(int)) ; 
    cudaMalloc (&nodePtrs_cuda, numNodePtrs * sizeof(int));
    cudaMemcpy(nodePtrs_cuda, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);

    int* currLevelNodes_cuda = (int*)malloc( numCurrLevelNodes * sizeof(int)) ; 
    cudaMalloc (&currLevelNodes_cuda, numCurrLevelNodes * sizeof(int));
    cudaMemcpy(currLevelNodes_cuda, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

    int* nodeNeighbors_cuda = (int*)malloc( numTotalNeighbors_h * sizeof(int)) ; 
    cudaMalloc (&nodeNeighbors_cuda, numTotalNeighbors_h * sizeof(int));
    cudaMemcpy(nodeNeighbors_cuda, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);

    int* nodeVisited_cuda = (int*)malloc( numNodesSize) ; 
    cudaMalloc (&nodeVisited_cuda, numNodesSize);
    cudaMemcpy(nodeVisited_cuda, nodeVisited_h,numNodesSize, cudaMemcpyHostToDevice);

    int* nodeGate_cuda = (int*)malloc( numNodesSize) ; 
    cudaMalloc (&nodeGate_cuda, numNodesSize);
    cudaMemcpy(nodeGate_cuda, nodeGate_h, numNodesSize, cudaMemcpyHostToDevice);

    int* nodeInput_cuda = (int*)malloc( numNodesSize) ; 
    cudaMalloc (&nodeInput_cuda, numNodesSize);
    cudaMemcpy(nodeInput_cuda, nodeInput_h, numNodesSize, cudaMemcpyHostToDevice);

    int* nodeOutput_cuda = (int*)malloc(numNodesSize) ; 
    cudaMalloc (&nodeOutput_cuda, numNodesSize);
    cudaMemcpy(nodeOutput_cuda, nodeOutput_h, numNodesSize, cudaMemcpyHostToDevice);

    // kernel call
    global_queuing_kernel <<< numBlocks, blockSize >>> (blockSize * numBlocks, numNodes, nodePtrs_cuda, currLevelNodes_cuda, nodeNeighbors_cuda, nodeVisited_cuda, nodeGate_cuda, nodeInput_cuda, nodeOutput_cuda);

    checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
    checkCudaErr(cudaGetLastError(), "GPU");

    cudaMemcpyFromSymbol(&numNextLevelNodes_h, numNextLevelNodes, sizeof(int), 0, cudaMemcpyDeviceToHost);
    checkCudaErr(cudaMemcpyFromSymbol(nextLevelNodes_h,globalQueue, numNodesSize), "Copying");

    int *outputBuffer;
    outputBuffer = (int*)malloc( numNodesSize); 
    checkCudaErr(cudaMemcpy(outputBuffer, nodeOutput_cuda, numNodesSize, cudaMemcpyDeviceToHost), "Copying");

    // write node output file
    FILE *nodeOutputFile = fopen(nodeOutput_fileName, "w");
    int counter = 0;
    fprintf(nodeOutputFile,"%d\n",numNodes);

    while (counter < numNodes) {
        fprintf(nodeOutputFile,"%d\n",(outputBuffer[counter]));
        counter++;
    }
    
    fclose(nodeOutputFile);
    
    // write next level output file
    FILE *nextLevelOutputFile = fopen(nextLevelNodes_fileName, "w");
    counter = 0;
    fprintf(nextLevelOutputFile,"%d\n",numNextLevelNodes_h);
    
    while (counter < numNextLevelNodes_h) {
        fprintf(nextLevelOutputFile,"%d\n",(nextLevelNodes_h[counter]));
        counter++;
    }
    
    fclose(nextLevelOutputFile);

    // free variables
    free(nodePtrs_h);
    free(nodeNeighbors_h);
    free(nodeVisited_h);
    free(currLevelNodes_h);
    free(nodeGate_h);
    free(nodeInput_h);
    free(nodeOutput_h);

    // free cuda variables
    cudaFree(currLevelNodes_cuda);
    cudaFree(nodeNeighbors_cuda);
    cudaFree(nodePtrs_cuda);
    cudaFree(nodeVisited_cuda);
    cudaFree(nodeInput_cuda);
    cudaFree(nodeOutput_cuda);
    cudaFree(nodeGate_cuda);
}
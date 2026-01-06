#include "CudaCompute.h"

#include <device_launch_parameters.h>
#include <cstdio>
#include <ctime>


// ################
// DEVICE
// ################



// calculating the sum of elements located within radius R
__device__ __forceinline__
float calculateElement(float* TAB, int N, int R, int x, int y)
{
    float sum = 0.0f;

    for (int i = x - R; i <= x + R; i++) {
        for (int j = y - R; j <= y + R; j++) {
            sum += TAB[i * N + j];
        }
    }

    return sum;
}


//copying the required memory fragment to shared memory
__device__ __forceinline__
void copyToShared2D(float* TAB, float* sharedTile, int N, int localTabXDim, int tileStartRow, int tileStartCol, int tileEndRow, int tileEndCol
)
{
    int linearThreadId = threadIdx.x * blockDim.y + threadIdx.y;
    int blockStride = blockDim.x * blockDim.y;

    //number of elements to copy
    int elementsToCopy =(tileEndRow * N + tileEndCol) -(tileStartRow * N + tileStartCol) + 1;

     //we start copying with a gap from the right side, equal to the gap from the beginning of the thread block we are servicing
    for (int i = linearThreadId; i < elementsToCopy; i += blockStride) {

        int index = i + tileStartCol;
        sharedTile[index] = TAB[tileStartRow * N + index];
    }

    __syncthreads();
}

// ################
// KERNEL
// ################

__global__
void addKernel(float* TAB, float* OUT, int M, int N, int R, int k, int threadsNum, int localTabXDim)
{
    // localId is calculated in wrong way, it is not coalescing and there is a problem with bank conflicts
    // subsequent threads are not allocated to subsequent memory cells
    int localId = threadIdx.x * blockDim.y + threadIdx.y;
    int blockThreads = blockDim.x * blockDim.y;
    int globalId = blockIdx.x * blockThreads + localId;

    extern __shared__ float sharedTile[];

    for (int i = 0; i < k; i++) {
        int outputIndex = i * threadsNum + globalId;

        if (outputIndex >= M * M) continue;

        //calculate the coordinates of the beginning and end of the thread block on a two-dimensional plane
        int tileStartRow = (outputIndex - localId) / M;
        int tileStartCol = (outputIndex - localId) % M;

        int tileEndRow = (blockThreads + outputIndex - localId - 1) / M + 2 * R;
        int tileEndCol = (blockThreads + outputIndex - localId - 1) % M + 2 * R;

        copyToShared2D(TAB, sharedTile, N, localTabXDim, tileStartRow, tileStartCol, tileEndRow, tileEndCol);

        //shared array data cells corresponding to the thread
        int sharedRow = (localId + tileStartCol) / M + R;
        int sharedCol = outputIndex % M + R;

        OUT[outputIndex] = calculateElement(sharedTile, N, R, sharedRow, sharedCol);
    }
}

// ################
// HOST
// ################

cudaError_t addWithCuda(float *TAB, float *OUT, int& N, int& M, int& R, int& BS, int& k)
{
    float *dev_tab = nullptr;
    float *dev_out = nullptr;

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    clock_t startt, stopt;
    float msecTotal = 0.0f;
    double gigaFlops = 0.0;

    dim3 block, grid;
    int threadsNum;

    size_t shmem_size = 0; // shared array size
    int shmem_rows = (M + BS * BS - 1) / M; //shared array row numbers

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_tab, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, M * M * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_tab, TAB, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

     cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed");
        goto Error;
    }

    startt = clock();

    block = dim3(BS, BS);
    grid = dim3((M * M + BS * BS * k - 1) / (BS * BS * k)); // formula is M*M / BS*BS*k, but it is ceil
    threadsNum = grid.x * block.x * block.y;

    //We add a spare row due to the difficulty of predicting the shape/distribution of the thread block 
    //relative to the corresponding thread cell in the array to be counted.
    //The smalest possible shered array is size is one row, so N elements.
    shmem_rows = shmem_rows + 2 * R + 1; 
    shmem_size = shmem_rows * N * sizeof(float);

    addKernel<<<grid, block, shmem_size>>>(dev_tab, dev_out, M, N, R, k, threadsNum, shmem_rows);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        if (cudaStatus == cudaErrorLaunchOutOfResources) {
            fprintf(stderr, "addKernel launch failed: przekroczono dostępną pamięć lub zasoby GPU (cudaErrorLaunchOutOfResources)!\n");
        } else {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }
    goto Error;
    }

  

    cudaStatus = cudaEventRecord(stop, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed");
        goto Error;
    }

    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventSynchronize(stop) failed");
        goto Error;
    }
    stopt = clock();

    printf("Czas przetwarzania wynosi %.3f msekund\n", ((double)(stopt - startt)));

    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec \n", gigaFlops, msecTotal);

    cudaStatus = cudaMemcpy(OUT, dev_out, M * M * sizeof(float), cudaMemcpyDeviceToHost);

Error:
    cudaFree(dev_tab);
    cudaFree(dev_out);

    return cudaStatus;
}

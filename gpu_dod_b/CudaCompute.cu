#include "CudaCompute.h"

#include <device_launch_parameters.h>
#include <cstdio>
#include <ctime>

// ##################
// DEVIC
// ##################


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

// ##################
// KERNEL
// ##################

__global__
void addKernel(float* TAB, float* OUT, int M, int N, int R, int k, int threadsNum)
{
    // localId is calculated in wrong way, it is not coalescing
    // subsequent threads are not allocated to subsequent memory cells
    int localId = threadIdx.y * blockDim.x + threadIdx.x;
    int globalId = blockIdx.x + localId * gridDim.x;

    for (int i = 0; i < k; i++) {
        int finalIndex = globalId * k + i;

        if (finalIndex >= M * M) continue;

        // calculating virtual 2d indexes
        int x = finalIndex / M + R;
        int y = finalIndex % M + R;

        OUT[finalIndex] = calculateElement(TAB, N, R, x, y);
    }
}

// ##################
// HOST
// ##################

cudaError_t addWithCuda(
    float* TAB,
    float* OUT,
    int& N,
    int& M,
    int& R,
    int& BS,
    int& k
) {
    float* dev_tab = nullptr;
    float* dev_out = nullptr;

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    clock_t startt, stopt;
    float msecTotal = 0.0f;
    double gigaFlops = 0.0;

    dim3 block, grid;
    int threadsNum;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_tab, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_out, M * M * sizeof(float));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_tab, TAB, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    startt = clock();

    block = dim3(BS, BS);
    grid = dim3((M * M + BS * BS * k - 1) / (BS * BS * k)); // formula is M*M / BS*BS*k, but it is ceil
    threadsNum = grid.x * block.x * block.y;

    addKernel<<<grid, block>>>(dev_tab, dev_out, M, N, R, k, threadsNum);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) goto Error;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    stopt = clock();

    printf("Czas przetwarzania wynosi %.3f ms\n",
           (double)(stopt - startt));

    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Performance= %.2f GFlop/s, Time= %.3f ms\n",
           gigaFlops, msecTotal);

    cudaStatus = cudaMemcpy(OUT, dev_out, M * M * sizeof(float), cudaMemcpyDeviceToHost);

Error:
    cudaFree(dev_tab);
    cudaFree(dev_out);
    return cudaStatus;
}

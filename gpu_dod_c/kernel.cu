#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <cstdlib>

using namespace std;
cudaError_t addWithCuda(
    float* TAB,
    float* OUT,
    int& N,
    int& M,
    int& R,
    int& BS,
    int& k
);


__device__ __forceinline__ float calculateElement(float* TAB, int N, int R, int x, int y)
{
    float sum = 0.0f;
    for (int i = x - R; i <= x + R; i++) {
        for (int j = y - R; j <= y + R; j++) {
            sum += TAB[i * N + j];
        }
    }
    return sum;
}

__device__ __forceinline__ void copyToShared2D(float* TAB, float* localTAB, int N, int localTab_x, int localTab_y, int startX, int startY)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;
    int total = localTab_x * localTab_y;

    for (int i = tid; i < total; i += stride) {
        int r = i / localTab_x;
        int c = i % localTab_x;
        localTAB[r * localTab_x + c] = TAB[(startX + r) * N + (startY + c)];
    }

    __syncthreads(); 
}

__global__ void addKernel(float* TAB, float* OUT, int M, int N, int R, int k, int threadsNum, int localTab_x, int localTab_y)
{
    int localId = threadIdx.y * blockDim.x + threadIdx.x;
    int blockThreads = blockDim.x * blockDim.y;
    int globalId = blockIdx.x * blockThreads + localId;

    extern __shared__ float localTAB[];


    for (int i = 0; i < k; i++) {
        int finalIndex = i * threadsNum + globalId;

        if (finalIndex >= M * M) continue;

        int startX = (blockIdx.x * blockThreads) / M;
        int startY = (blockIdx.x * blockThreads) % M;

        copyToShared2D(TAB, localTAB, N, localTab_x, localTab_y, startX, startY);

    
        int localX = localId / localTab_x + R;
        int localY = localId % localTab_x + R;
        OUT[finalIndex] = calculateElement(localTAB, localTab_x, R, localX, localY);
    }
}



bool loadDataFromFile(const string& fileName, int& N, int& R, float**& matrix, float*& data)
{
    ifstream file(fileName);
    if (!file.is_open()) {
        cerr << "Cannot open file!" << endl;
        return false;
    }

    file >> N >> R;

    data = new float[N * N];
    matrix = new float* [N];
    for (int i = 0; i < N; i++)
        matrix[i] = data + i * N;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            file >> matrix[i][j];

    file.close();
    return true;
}

bool saveMatrixToFile(const string& fileName, float** matrix, int N)
{
    ofstream file(fileName);
    if (!file.is_open()) {
        cerr << "Cannot open output file!" << endl;
        return false;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            file << matrix[i][j];
            if (j < N - 1) file << " ";
        }
        file << "\n";
    }

    file.close();
    return true;
}


int main(int argc, char* argv[])
{
    int N, R, BS, k;

    k = std::atoi(argv[1]);
    BS = std::atoi(argv[2]);

    float* TAB_o = nullptr;
    float* OUT_o = nullptr;
    float** TAB = nullptr;
    float** OUT = nullptr;

    if (!loadDataFromFile("dane.txt", N, R, TAB, TAB_o))
        return 1;

    int M = N - 2 * R;
    if (M <= 0) {
        cerr << "R too large!" << endl;
        return 1;
    }

    OUT_o = new float[M * M];
    OUT = new float* [M];
    for (int i = 0; i < M; i++)
        OUT[i] = OUT_o + i * M;

    cudaError_t status = addWithCuda(TAB_o, OUT_o, N, M, R, BS, k);
    if (status != cudaSuccess) {
        cerr << "CUDA error!" << endl;
        return 1;
    }

    saveMatrixToFile("odp.txt", OUT, M);

    delete[] TAB_o;
    delete[] TAB;
    delete[] OUT_o;
    delete[] OUT;

    if (status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    getchar();
    return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *TAB, float *OUT, int& N, int& M, int& R, int& BS, int& k)
{
    float *dev_tab = 0;
    float *dev_out = 0;

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    clock_t startt, stopt;
    float msecTotal = 0.0f;
    double gigaFlops = 0;

    dim3 block, grid;
    int threadsNum;
    
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
    startt = clock();//jeszcze obliczenia się nie zaczęły

    block = dim3(BS, BS);
    grid = dim3((M * M + BS * BS * k - 1) / (BS * BS * k));
    threadsNum = grid.x * block.x * block.y;


    size_t shmem_size =0;

    int shmem_x = 0;
    int shmem_y = 0; 
    int shmem_rows = (M + BS*BS - 1) / (BS*BS);

    if(shmem_rows< 2){
        shmem_x = 1 + 2 * R;
        shmem_y = BS * BS + 2 * R;
    }else{
        shmem_x = shmem_rows + 2 * R;
        shmem_y = N;
    }

    shmem_size = shem_x * shem_y * sizeof(float);
    
    addKernel<<<grid, block, shmem_size>>>(dev_tab, dev_out, M, N, R, k, threadsNum, shmem_x, shmem_y);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    stopt = clock();//obliczenia już zakończone

    printf("Czas przetwarzania wynosi %.3f msekund\n", ((double)(stopt - startt)));

    
    cudaEventElapsedTime(&msecTotal, start, stop);
    //obliczenia predkości obliczeń 
    gigaFlops = 0;//  (liczba operacji arytmetycznych) / (msecTotal / 1000.0f); */
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec \n",
        gigaFlops, msecTotal);
   
    cudaStatus = cudaMemcpy(OUT, dev_out, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_tab);
    cudaFree(dev_out);
    
    return cudaStatus;
}


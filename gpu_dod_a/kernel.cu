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


__global__ void addKernel(float* TAB, float* OUT, int M, int N, int R, int k, int threadsNum)
{
    int localId = threadIdx.y * blockDim.x + threadIdx.x;

    int globalId = blockIdx.x * (blockDim.x * blockDim.y) + localId;

    for (int i = 0; i < k; i++) {
        int finalIndex = threadsNum * i + globalId;

        if (finalIndex >= M * M) continue;

        int x = finalIndex / M + R;
        int y = finalIndex % M + R;
        OUT[finalIndex] = calculateElement(TAB, N, R, x, y);
        
    }
}


bool loadDataFromFile(const string& fileName, int& N, float**& matrix, float*& data)
{
    ifstream file(fileName);
    if (!file.is_open()) {
        cerr << "Cannot open file!" << endl;
        return false;
    }

    file >> N;

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

    string filename = argv[1];

    R = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
    BS = std::atoi(argv[4]);

    float* TAB_o = nullptr;
    float* OUT_o = nullptr;
    float** TAB = nullptr;
    float** OUT = nullptr;

    if (!loadDataFromFile(filename, N, TAB, TAB_o))
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

    addKernel<<<grid, block>>>(dev_tab, dev_out, M, N, R, k, threadsNum);
    
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


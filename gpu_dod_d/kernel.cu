
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
            //if(x == 1 && y == 1) printf("%d z %d z %f z %d \n ", x, y, TAB[i * N + j], i*N+j);
        }
    }
	// printf("%d %d sum%f \n", x, y, sum);
    return sum;
}

__device__ __forceinline__ void copyToShared2D(float* TAB, float* localTAB, int N, int localTabXDim, int startX, int startY, int endX, int endY)
{
    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    int stride = blockDim.x * blockDim.y;
    int total = (endX * N + endY) - (startX * N + startY) + 1;

    for (int i = tid; i < total ; i += stride) {

	int index = i + startY;
	 localTAB[index]= TAB[startX * N + index];
//	if(startX == 1 && startY == 10)	printf("%d, %d, %d, %d \n", index/N, index%N,(startX * N + index)/N, (startX * N + index)%N );
    }

    __syncthreads(); 
}

__global__ void addKernel(float* TAB, float* OUT, int M, int N, int R, int k, int threadsNum, int localTabXDim)
{
    int localId = threadIdx.x * blockDim.y + threadIdx.y;
    int blockThreads = blockDim.x * blockDim.y;

    int globalId = blockIdx.x + localId * gridDim.x;
    

    extern __shared__ float localTAB[];


    for (int i = 0; i < k; i++) {
         int finalIndex = globalId * k + i;

        if (finalIndex >= M * M) continue;

        int startX = (finalIndex - localId ) / M;
        int startY = (finalIndex - localId) % M;
	    int endX = (blockThreads+finalIndex - localId-1) / M + 2 * R;
        int endY = (blockThreads+finalIndex - localId-1) % M + 2 * R;

        copyToShared2D(TAB, localTAB, N, localTabXDim, startX, startY, endX, endY);

        int localX = (localId+startY) / M + R;
        int localY = finalIndex % M + R;
//	printf("%d, %d, %d \n", finalIndex, localId, i);
//	printf("%d,, %d,, BB%d \n", localX, localY, finalIndex);
        OUT[finalIndex] = calculateElement(localTAB, N, R, localX, localY);
//	printf("%d o %d o CC%d o %f \n", localX, localY, finalIndex, OUT[finalIndex]);
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
    for (int i = 0; i < N; i++){
        matrix[i] = data + i * N;}

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            file >> matrix[i][j];

    file.close();
    return true;
}

bool saveMatrixToFile(const string& fileName, float** matrix, int N, int M)
{
    ofstream file(fileName);
    if (!file.is_open()) {
        cerr << "Cannot open output file!" << endl;
        return false;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
//	    printf("%d, %d, %f \n", i, j, matrix[i][j]);
            file << matrix[i][j];
            if (j < M - 1) file << " ";
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

    saveMatrixToFile("odp.txt", OUT, M, M);

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

    size_t shmem_size =0;
    int shmem_rows = (M + BS*BS - 1) / M;
    
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




    shmem_rows = shmem_rows + 2 * R + 1;
    shmem_size = shmem_rows * N  * sizeof(float);

    addKernel<<<grid, block, shmem_size>>>(dev_tab, dev_out, M, N, R, k, threadsNum, shmem_rows);

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




#include <iostream>
#include <cstdlib>
#include <string>

#include "FileManager.h"
#include "CudaCompute.h"

#include <cuda_runtime.h>



using namespace std;




int main(int argc, char* argv[])
{
    if (argc < 5) {
        cerr << "Usage: program <file> <R> <k> <BS>" << endl;
        return 1;
    }

    // ===== Argumenty =====
    string filename = argv[1];
    int R  = std::atoi(argv[2]);
    int k  = std::atoi(argv[3]);
    int BS = std::atoi(argv[4]);

    // ===== Dane wejściowe =====
    int N;
    float* TAB_o = nullptr;
    float** TAB = nullptr;

    if (!FileManager::loadMatrix(filename, N, TAB, TAB_o))
        return 1;

    int M = N - 2 * R;
    if (M <= 0) {
        cerr << "R too large!" << endl;
        return 1;
    }

    // ===== Dane wyjściowe =====
    float* OUT_o = new float[M * M];
    float** OUT = new float*[M];
    for (int i = 0; i < M; i++)
        OUT[i] = OUT_o + i * M;

    // ===== CUDA =====
    cudaError_t status = addWithCuda(TAB_o, OUT_o, N, M, R, BS, k);
    if (status != cudaSuccess) {
        cerr << "CUDA error!" << endl;
        return 1;
    }

    // ===== Zapis wyniku =====
    FileManager::saveMatrix("odp.txt", OUT, M);

    // ===== Sprzątanie =====
    delete[] TAB_o;
    delete[] TAB;
    delete[] OUT_o;
    delete[] OUT;

    cudaDeviceReset();
    return 0;
}





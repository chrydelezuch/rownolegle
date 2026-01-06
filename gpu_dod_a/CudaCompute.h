#ifndef CUDA_COMPUTE_H
#define CUDA_COMPUTE_H

#include <cuda_runtime.h>

cudaError_t addWithCuda(
    float* TAB,
    float* OUT,
    int& N,
    int& M,
    int& R,
    int& BS,
    int& k
);

#endif // CUDA_COMPUTE_H

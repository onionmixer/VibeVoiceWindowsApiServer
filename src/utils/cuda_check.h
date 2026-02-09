#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s (code %d)\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err), (int)err);    \
            return false;                                                      \
        }                                                                      \
    } while (0)

#pragma once
#include <cuda_runtime.h>
#include "utils/logger.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            LOG_ERROR("CUDA", "error at %s:%d: %s (code %d)",                 \
                      __FILE__, __LINE__, cudaGetErrorString(err), (int)err);  \
            return false;                                                      \
        }                                                                      \
    } while (0)

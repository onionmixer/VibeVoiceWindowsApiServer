#include "inference/gpu_ops.h"
#include <cstdio>

// ── CUDA kernel implementations ──

// Embedding lookup: each thread copies one embedding vector
__global__ void embeddingLookupKernel(const __half* table, const int32_t* ids,
                                       __half* output, int numIds, int embedDim) {
    int idx = blockIdx.x;       // which id
    int elem = threadIdx.x;     // which element in the embedding vector
    if (idx >= numIds || elem >= embedDim) return;

    // Handle embedDim > blockDim.x with a loop
    for (int e = elem; e < embedDim; e += blockDim.x) {
        int tokenId = ids[idx];
        output[idx * embedDim + e] = table[tokenId * embedDim + e];
    }
}

__global__ void vectorAddKernel(const __half* a, const __half* b, __half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = __hadd(a[i], b[i]);
}

__global__ void rmsNormKernel(const __half* input, const __half* weight,
                               __half* output, int n, float eps) {
    // Single block: compute RMS norm for a vector of size n
    // Shared memory for partial sums
    extern __shared__ float sharedMem[];

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = __half2float(input[i]);
        localSum += val * val;
    }

    // Reduce within block
    sharedMem[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(sharedMem[0] / (float)n + eps);
    float invRms = 1.0f / rms;

    // Apply normalization and weight
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = __half2float(input[i]);
        float w = __half2float(weight[i]);
        output[i] = __float2half(val * invRms * w);
    }
}

__global__ void addBiasKernel(const __half* bias, __half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = __hadd(output[i], bias[i]);
}

__global__ void reluKernel(const __half* input, __half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half zero = __float2half(0.0f);
    output[i] = __hgt(input[i], zero) ? input[i] : zero;
}

__global__ void sigmoidKernel(const __half* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = __half2float(input[i]);
    output[i] = 1.0f / (1.0f + expf(-val));
}

__global__ void scaleAndBiasKernel(const __half* input, float scale, float bias,
                                    __half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = __half2float(input[i]);
    output[i] = __float2half(val / scale - bias);
}

__global__ void reverseScaleAndBiasKernel(const __half* input, float scale, float bias,
                                           __half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = __half2float(input[i]);
    output[i] = __float2half((val + bias) * scale);
}

__global__ void cfgBlendKernel(const __half* cond, const __half* uncond, float scale,
                                __half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = __half2float(cond[i]);
    float u = __half2float(uncond[i]);
    output[i] = __float2half(u + scale * (c - u));
}

// Replace masked embedding rows: embeds[maskIndices[t]] = speechFeatures[t]
__global__ void replaceMaskedEmbedsKernel(__half* embeds, const __half* speechFeatures,
                                           const int32_t* maskIndices, int T, int H) {
    int t = blockIdx.x;     // which speech token
    if (t >= T) return;
    int dstRow = maskIndices[t];
    for (int e = threadIdx.x; e < H; e += blockDim.x) {
        embeds[dstRow * H + e] = speechFeatures[t * H + e];
    }
}

// Batched RMSNorm: one block per row
__global__ void rmsNormBatchedKernel(const __half* input, const __half* weight,
                                      __half* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float sharedMem[];

    const __half* inRow = input + row * N;
    __half* outRow = output + row * N;

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = __half2float(inRow[i]);
        localSum += val * val;
    }

    sharedMem[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(sharedMem[0] / (float)N + eps);
    float invRms = 1.0f / rms;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = __half2float(inRow[i]);
        float w = __half2float(weight[i]);
        outRow[i] = __float2half(val * invRms * w);
    }
}

// Add bias to each row: output[row, i] += bias[i]
__global__ void addBiasBatchedKernel(const __half* bias, __half* output, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int col = idx % N;
    output[idx] = __hadd(output[idx], bias[col]);
}

__global__ void halfToFloatKernel(const __half* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = __half2float(input[i]);
}

__global__ void floatToHalfKernel(const float* input, __half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = __float2half(input[i]);
}

// ── FP32 kernels for connector pipeline ──

__global__ void vectorAddF32Kernel(const float* a, const float* b, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = a[i] + b[i];
}

__global__ void addBiasF32Kernel(const float* bias, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] += bias[i];
}

__global__ void rmsNormF32Kernel(const float* input, const float* weight,
                                  float* output, int n, float eps) {
    extern __shared__ float sharedMem[];

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = input[i];
        localSum += val * val;
    }

    sharedMem[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(sharedMem[0] / (float)n + eps);
    float invRms = 1.0f / rms;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = input[i] * invRms * weight[i];
    }
}

// ── Namespace implementation ──

namespace GpuOps {

static constexpr int kBlockSize = 256;
static int gridSize(int n) { return (n + kBlockSize - 1) / kBlockSize; }

void embeddingLookup(const __half* table, const int32_t* ids, __half* output,
                     int numIds, int embedDim, cudaStream_t stream) {
    int threads = (embedDim < 1024) ? embedDim : 1024;
    embeddingLookupKernel<<<numIds, threads, 0, stream>>>(table, ids, output, numIds, embedDim);
}

void vectorAdd(const __half* a, const __half* b, __half* output,
               int n, cudaStream_t stream) {
    vectorAddKernel<<<gridSize(n), kBlockSize, 0, stream>>>(a, b, output, n);
}

void rmsNorm(const __half* input, const __half* weight, __half* output,
             int n, float eps, cudaStream_t stream) {
    int threads = (n < 1024) ? n : 1024;
    // Round up to power of 2 for reduction
    int reducThreads = 1;
    while (reducThreads < threads) reducThreads <<= 1;
    if (reducThreads > 1024) reducThreads = 1024;

    size_t sharedSize = reducThreads * sizeof(float);
    rmsNormKernel<<<1, reducThreads, sharedSize, stream>>>(input, weight, output, n, eps);
}

void linearForward(cublasHandle_t cublas, const __half* input,
                   const __half* weight, const __half* bias,
                   __half* output, int N, int K, cudaStream_t stream) {
    // output[1,N] = input[1,K] @ weight[N,K]^T
    // Using cuBLAS: C = alpha * A * B + beta * C
    // We want: output = weight^T @ input^T  =>  (N,1) = (N,K) @ (K,1)
    // cuBLAS column-major: Cgemv or Cgemm
    // Using cublasHgemm: C[N,1] = weight[N,K] * input_col[K,1]
    // In column-major: weight is (K,N) stored col-major = (N,K) row-major. OK.
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasSetStream(cublas, stream);

    // weight is [N, K] in row-major = [K, N] in column-major
    // input is [1, K] row-major = [K, 1] column-major
    // output is [1, N] row-major = [N, 1] column-major
    // C(N,1) = A(N,K) * B(K,1)  in col-major: A stored as (K,N) with transpose
    // cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, alpha, weight, K, input, K, beta, output, N)
    cublasHgemm(cublas,
                CUBLAS_OP_T,    // A (weight) transposed: (K,N)^T = (N,K)
                CUBLAS_OP_N,    // B (input) not transposed: (K,1)
                N, 1, K,       // m, n, k
                &alpha,
                weight, K,     // lda = K (col-major storage of weight)
                input, K,      // ldb = K
                &beta,
                output, N);    // ldc = N

    // Add bias if present
    if (bias) {
        int blocks = gridSize(N);
        addBiasKernel<<<blocks, kBlockSize, 0, stream>>>(bias, output, N);
    }
}

void relu(const __half* input, __half* output, int n, cudaStream_t stream) {
    reluKernel<<<gridSize(n), kBlockSize, 0, stream>>>(input, output, n);
}

void sigmoid(const __half* input, float* output, int n, cudaStream_t stream) {
    sigmoidKernel<<<gridSize(n), kBlockSize, 0, stream>>>(input, output, n);
}

void scaleAndBias(const __half* input, float scale, float bias,
                  __half* output, int n, cudaStream_t stream) {
    scaleAndBiasKernel<<<gridSize(n), kBlockSize, 0, stream>>>(input, scale, bias, output, n);
}

void reverseScaleAndBias(const __half* input, float scale, float bias,
                         __half* output, int n, cudaStream_t stream) {
    reverseScaleAndBiasKernel<<<gridSize(n), kBlockSize, 0, stream>>>(input, scale, bias, output, n);
}

void cfgBlend(const __half* cond, const __half* uncond, float scale,
              __half* output, int n, cudaStream_t stream) {
    cfgBlendKernel<<<gridSize(n), kBlockSize, 0, stream>>>(cond, uncond, scale, output, n);
}

void replaceMaskedEmbeds(__half* embeds, const __half* speechFeatures,
                         const int32_t* maskIndices, int T, int H,
                         cudaStream_t stream) {
    if (T <= 0) return;
    int threads = (H < 1024) ? H : 1024;
    replaceMaskedEmbedsKernel<<<T, threads, 0, stream>>>(embeds, speechFeatures, maskIndices, T, H);
}

void linearForwardBatch(cublasHandle_t cublas, const __half* A,
                        const __half* B, const __half* bias,
                        __half* C, int M, int N, int K,
                        cudaStream_t stream) {
    // C[M,N] = A[M,K] @ B[N,K]^T
    // cuBLAS column-major: C_col[N,M] = B_col[N,K] * A_col[K,M]
    // B is [N,K] row-major = [K,N] col-major → need transpose
    // A is [M,K] row-major = [K,M] col-major → no transpose needed
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasSetStream(cublas, stream);

    cublasHgemm(cublas,
                CUBLAS_OP_T,    // B transposed: (K,N)^T = (N,K)
                CUBLAS_OP_N,    // A not transposed: (K,M)
                N, M, K,       // m, n, k  (output is N x M in col-major)
                &alpha,
                B, K,          // lda = K
                A, K,          // ldb = K
                &beta,
                C, N);         // ldc = N

    if (bias) {
        int total = M * N;
        addBiasBatchedKernel<<<gridSize(total), kBlockSize, 0, stream>>>(bias, C, M, N);
    }
}

void rmsNormBatched(const __half* input, const __half* weight, __half* output,
                    int M, int N, float eps, cudaStream_t stream) {
    int threads = (N < 1024) ? N : 1024;
    // Round up to power of 2 for reduction
    int reducThreads = 1;
    while (reducThreads < threads) reducThreads <<= 1;
    if (reducThreads > 1024) reducThreads = 1024;

    size_t sharedSize = reducThreads * sizeof(float);
    rmsNormBatchedKernel<<<M, reducThreads, sharedSize, stream>>>(input, weight, output, M, N, eps);
}

void halfToFloat(const __half* input, float* output, int n, cudaStream_t stream) {
    halfToFloatKernel<<<gridSize(n), kBlockSize, 0, stream>>>(input, output, n);
}

void floatToHalf(const float* input, __half* output, int n, cudaStream_t stream) {
    floatToHalfKernel<<<gridSize(n), kBlockSize, 0, stream>>>(input, output, n);
}

// ── FP32 connector operations (matching Python reference precision) ──

void linearForwardF32(cublasHandle_t cublas, const float* input,
                      const float* weight, const float* bias,
                      float* output, int N, int K, cudaStream_t stream) {
    // output[1,N] = input[1,K] @ weight[N,K]^T  (all fp32)
    // Same logic as fp16 version but using cublasSgemm
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSetStream(cublas, stream);

    // weight is [N, K] in row-major = [K, N] in column-major
    // C(N,1) = A(N,K) * B(K,1) in col-major: A stored as (K,N) with transpose
    cublasSgemm(cublas,
                CUBLAS_OP_T,    // A (weight) transposed
                CUBLAS_OP_N,    // B (input) not transposed
                N, 1, K,
                &alpha,
                weight, K,      // lda = K
                input, K,       // ldb = K
                &beta,
                output, N);     // ldc = N

    if (bias) {
        int blocks = gridSize(N);
        addBiasF32Kernel<<<blocks, kBlockSize, 0, stream>>>(bias, output, N);
    }
}

void rmsNormF32(const float* input, const float* weight, float* output,
                int n, float eps, cudaStream_t stream) {
    int threads = (n < 1024) ? n : 1024;
    int reducThreads = 1;
    while (reducThreads < threads) reducThreads <<= 1;
    if (reducThreads > 1024) reducThreads = 1024;

    size_t sharedSize = reducThreads * sizeof(float);
    rmsNormF32Kernel<<<1, reducThreads, sharedSize, stream>>>(input, weight, output, n, eps);
}

void vectorAddF32(const float* a, const float* b, float* output,
                  int n, cudaStream_t stream) {
    vectorAddF32Kernel<<<gridSize(n), kBlockSize, 0, stream>>>(a, b, output, n);
}

} // namespace GpuOps

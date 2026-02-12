#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace GpuOps {

// Embedding lookup: output[i] = table[ids[i]] (fp16)
// table: [num_embeddings, embed_dim], ids: [num_ids], output: [num_ids, embed_dim]
void embeddingLookup(const __half* table, const int32_t* ids, __half* output,
                     int numIds, int embedDim, cudaStream_t stream);

// output = a + b (element-wise fp16 vector add)
void vectorAdd(const __half* a, const __half* b, __half* output,
               int n, cudaStream_t stream);

// RMSNorm: output = normalize(input) * weight
// input/output: [n] fp16, weight: [n] fp16, eps: float
void rmsNorm(const __half* input, const __half* weight, __half* output,
             int n, float eps, cudaStream_t stream);

// Linear (cuBLAS GEMV): output = input @ weight^T + bias
// input: [1, K] fp16, weight: [N, K] fp16, bias: [N] fp16 (nullable), output: [1, N] fp16
void linearForward(cublasHandle_t cublas, const __half* input,
                   const __half* weight, const __half* bias,
                   __half* output, int N, int K, cudaStream_t stream);

// ReLU: output = max(0, input) (fp16)
void relu(const __half* input, __half* output, int n, cudaStream_t stream);

// Sigmoid: output = 1/(1+exp(-input)) (fp16 -> fp32 result)
void sigmoid(const __half* input, float* output, int n, cudaStream_t stream);

// Scale+bias: output = input / scale - bias (fp16, scalar scale/bias in fp32)
void scaleAndBias(const __half* input, float scale, float bias,
                  __half* output, int n, cudaStream_t stream);

// Reverse scale: output = (input + bias) * scale (for 1.5B encoding)
void reverseScaleAndBias(const __half* input, float scale, float bias,
                         __half* output, int n, cudaStream_t stream);

// CFG blend: output = uncond + scale * (cond - uncond) (fp16)
void cfgBlend(const __half* cond, const __half* uncond, float scale,
              __half* output, int n, cudaStream_t stream);

// Replace masked embedding positions with speech features (in-place)
// embeds: [totalTokens, H] fp16 (modified in-place)
// speechFeatures: [T, H] fp16
// maskIndices: [T] int32 (indices into embeds where speech tokens go)
void replaceMaskedEmbeds(__half* embeds, const __half* speechFeatures,
                         const int32_t* maskIndices, int T, int H,
                         cudaStream_t stream);

// Batched Linear: C = A @ B^T + bias  (M rows)
// A: [M, K], B: [N, K], bias: [N] (nullable), C: [M, N]  all fp16
void linearForwardBatch(cublasHandle_t cublas, const __half* A,
                        const __half* B, const __half* bias,
                        __half* C, int M, int N, int K,
                        cudaStream_t stream);

// Batched RMSNorm: normalize each of M rows independently
// input/output: [M, N], weight: [N]  all fp16
void rmsNormBatched(const __half* input, const __half* weight, __half* output,
                    int M, int N, float eps, cudaStream_t stream);

// FP32 variants for connector pipeline (matching Python reference precision)
// Linear: output = input @ weight^T + bias  (all fp32)
void linearForwardF32(cublasHandle_t cublas, const float* input,
                      const float* weight, const float* bias,
                      float* output, int N, int K, cudaStream_t stream);

// RMSNorm: output = normalize(input) * weight  (all fp32)
void rmsNormF32(const float* input, const float* weight, float* output,
                int n, float eps, cudaStream_t stream);

// Vector add: output = a + b  (all fp32)
void vectorAddF32(const float* a, const float* b, float* output,
                  int n, cudaStream_t stream);

// fp16 <-> fp32 conversion
void halfToFloat(const __half* input, float* output, int n, cudaStream_t stream);
void floatToHalf(const float* input, __half* output, int n, cudaStream_t stream);

} // namespace GpuOps

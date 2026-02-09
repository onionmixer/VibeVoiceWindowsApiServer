#include "utils/cuda_buffer.h"
#include "utils/cuda_check.h"
#include <cstring>

CudaBuffer::CudaBuffer() = default;

CudaBuffer::~CudaBuffer() {
    free();
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), capacity_(other.capacity_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

bool CudaBuffer::resize(size_t sizeBytes) {
    if (sizeBytes <= capacity_) {
        size_ = sizeBytes;
        return true;
    }
    // Free old allocation
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
    }
    CUDA_CHECK(cudaMalloc(&ptr_, sizeBytes));
    capacity_ = sizeBytes;
    size_ = sizeBytes;
    return true;
}

void CudaBuffer::free() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
    size_ = 0;
    capacity_ = 0;
}

bool CudaBuffer::copyFromHost(const void* hostSrc, size_t bytes) {
    if (bytes > capacity_) return false;
    CUDA_CHECK(cudaMemcpy(ptr_, hostSrc, bytes, cudaMemcpyHostToDevice));
    size_ = bytes;
    return true;
}

bool CudaBuffer::copyToHost(void* hostDst, size_t bytes) const {
    if (bytes > size_) return false;
    CUDA_CHECK(cudaMemcpy(hostDst, ptr_, bytes, cudaMemcpyDeviceToHost));
    return true;
}

bool CudaBuffer::copyFromHostAsync(const void* hostSrc, size_t bytes, cudaStream_t stream) {
    if (bytes > capacity_) return false;
    CUDA_CHECK(cudaMemcpyAsync(ptr_, hostSrc, bytes, cudaMemcpyHostToDevice, stream));
    size_ = bytes;
    return true;
}

bool CudaBuffer::copyToHostAsync(void* hostDst, size_t bytes, cudaStream_t stream) const {
    if (bytes > size_) return false;
    CUDA_CHECK(cudaMemcpyAsync(hostDst, ptr_, bytes, cudaMemcpyDeviceToHost, stream));
    return true;
}

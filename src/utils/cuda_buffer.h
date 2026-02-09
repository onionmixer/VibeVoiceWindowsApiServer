#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

class CudaBuffer {
public:
    CudaBuffer();
    ~CudaBuffer();

    // Move-only
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Allocate / resize GPU memory. Only reallocates if capacity < sizeBytes.
    bool resize(size_t sizeBytes);

    // Free GPU memory explicitly.
    void free();

    // Synchronous host <-> device copies.
    bool copyFromHost(const void* hostSrc, size_t bytes);
    bool copyToHost(void* hostDst, size_t bytes) const;

    // Async copies with stream.
    bool copyFromHostAsync(const void* hostSrc, size_t bytes, cudaStream_t stream);
    bool copyToHostAsync(void* hostDst, size_t bytes, cudaStream_t stream) const;

    // Convenience: upload a std::vector to GPU.
    template <typename T>
    bool upload(const std::vector<T>& vec) {
        size_t bytes = vec.size() * sizeof(T);
        if (!resize(bytes)) return false;
        return copyFromHost(vec.data(), bytes);
    }

    // Convenience: download GPU data into a std::vector.
    template <typename T>
    bool download(std::vector<T>& vec) const {
        if (size_ == 0) { vec.clear(); return true; }
        vec.resize(size_ / sizeof(T));
        return copyToHost(vec.data(), size_);
    }

    // Typed pointer access.
    template <typename T>
    T* as() { return reinterpret_cast<T*>(ptr_); }

    template <typename T>
    const T* as() const { return reinterpret_cast<const T*>(ptr_); }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
};

#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// TensorRT logger singleton
class TRTLogger : public nvinfer1::ILogger {
public:
    static TRTLogger& instance();
    void log(Severity severity, const char* msg) noexcept override;
    void setVerbose(bool verbose);

private:
    TRTLogger() = default;
    bool verbose_ = false;
};

// I/O tensor metadata
struct TensorInfo {
    std::string name;
    nvinfer1::Dims dims;        // shape (may contain -1 for dynamic dims)
    nvinfer1::DataType dataType;
    nvinfer1::TensorIOMode ioMode;  // kINPUT or kOUTPUT

    // Total element count for a given concrete shape.
    int64_t elementCount(const nvinfer1::Dims& shape) const;
    // Byte size for a given concrete shape.
    int64_t byteSize(const nvinfer1::Dims& shape) const;
};

// TensorRT engine wrapper (load + execute)
class TRTEngine {
public:
    TRTEngine();
    ~TRTEngine();

    // Non-copyable, movable
    TRTEngine(TRTEngine&& other) noexcept;
    TRTEngine& operator=(TRTEngine&& other) noexcept;
    TRTEngine(const TRTEngine&) = delete;
    TRTEngine& operator=(const TRTEngine&) = delete;

    // Load engine from serialized .trt file.
    bool loadFromFile(const std::string& path);

    // Get tensor info by name.
    const TensorInfo* getTensorInfo(const std::string& name) const;

    // Get all I/O tensor infos.
    const std::vector<TensorInfo>& getTensors() const { return tensors_; }

    // Set dynamic input shape for a tensor.
    bool setInputShape(const std::string& name, const nvinfer1::Dims& dims);

    // Bind a GPU buffer to a tensor by name.
    bool setTensorAddress(const std::string& name, void* devicePtr);

    // Execute inference asynchronously on a CUDA stream.
    bool enqueueV3(cudaStream_t stream);

    // Query actual output tensor shape after execution (for dynamic shapes).
    nvinfer1::Dims getOutputShape(const std::string& name) const;

    // Check if engine is loaded.
    bool isLoaded() const { return engine_ != nullptr; }

private:
    void release();
    bool enumerateTensors();

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensorIndex_;
};

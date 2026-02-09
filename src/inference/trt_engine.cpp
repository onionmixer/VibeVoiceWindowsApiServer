#include "inference/trt_engine.h"
#include "utils/logger.h"
#include <fstream>
#include <vector>

// ── TRTLogger ──

TRTLogger& TRTLogger::instance() {
    static TRTLogger logger;
    return logger;
}

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                LOG_ERROR("TRT", "%s", msg);
                break;
            case Severity::kWARNING:
                LOG_WARN("TRT", "%s", msg);
                break;
            default:
                break;
        }
    } else if (verbose_) {
        LOG_DEBUG("TRT", "%s", msg);
    }
}

void TRTLogger::setVerbose(bool verbose) {
    verbose_ = verbose;
}

// ── TensorInfo ──

static int dataTypeSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 0;
    }
}

int64_t TensorInfo::elementCount(const nvinfer1::Dims& shape) const {
    int64_t count = 1;
    for (int i = 0; i < shape.nbDims; ++i) {
        count *= shape.d[i];
    }
    return count;
}

int64_t TensorInfo::byteSize(const nvinfer1::Dims& shape) const {
    return elementCount(shape) * dataTypeSize(dataType);
}

// ── TRTEngine ──

TRTEngine::TRTEngine() = default;

TRTEngine::~TRTEngine() {
    release();
}

TRTEngine::TRTEngine(TRTEngine&& other) noexcept
    : runtime_(other.runtime_), engine_(other.engine_), context_(other.context_),
      tensors_(std::move(other.tensors_)), tensorIndex_(std::move(other.tensorIndex_)) {
    other.runtime_ = nullptr;
    other.engine_ = nullptr;
    other.context_ = nullptr;
}

TRTEngine& TRTEngine::operator=(TRTEngine&& other) noexcept {
    if (this != &other) {
        release();
        runtime_ = other.runtime_;
        engine_ = other.engine_;
        context_ = other.context_;
        tensors_ = std::move(other.tensors_);
        tensorIndex_ = std::move(other.tensorIndex_);
        other.runtime_ = nullptr;
        other.engine_ = nullptr;
        other.context_ = nullptr;
    }
    return *this;
}

void TRTEngine::release() {
    // Destroy in reverse order of creation
    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_)  { engine_->destroy();  engine_ = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
    tensors_.clear();
    tensorIndex_.clear();
}

bool TRTEngine::loadFromFile(const std::string& path) {
    release();

    // Read serialized engine
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        LOG_ERROR("TRT", "Cannot open TRT engine: %s", path.c_str());
        return false;
    }
    size_t fileSize = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> engineData(fileSize);
    f.read(engineData.data(), fileSize);
    if (!f.good()) {
        LOG_ERROR("TRT", "Failed to read TRT engine file: %s", path.c_str());
        return false;
    }

    // Create runtime
    runtime_ = nvinfer1::createInferRuntime(TRTLogger::instance());
    if (!runtime_) {
        LOG_ERROR("TRT", "Failed to create TensorRT runtime");
        return false;
    }

    // Deserialize engine
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), fileSize);
    if (!engine_) {
        LOG_ERROR("TRT", "Failed to deserialize TRT engine: %s", path.c_str());
        return false;
    }

    // Create execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        LOG_ERROR("TRT", "Failed to create TRT execution context");
        return false;
    }

    // Enumerate I/O tensors
    if (!enumerateTensors()) {
        LOG_ERROR("TRT", "Failed to enumerate tensors");
        return false;
    }

    return true;
}

bool TRTEngine::enumerateTensors() {
    tensors_.clear();
    tensorIndex_.clear();

    int nbTensors = engine_->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (!name) continue;

        TensorInfo info;
        info.name = name;
        info.dims = engine_->getTensorShape(name);
        info.dataType = engine_->getTensorDataType(name);
        info.ioMode = engine_->getTensorIOMode(name);

        tensorIndex_[info.name] = tensors_.size();
        tensors_.push_back(std::move(info));
    }

    return true;
}

const TensorInfo* TRTEngine::getTensorInfo(const std::string& name) const {
    auto it = tensorIndex_.find(name);
    if (it == tensorIndex_.end()) return nullptr;
    return &tensors_[it->second];
}

bool TRTEngine::setInputShape(const std::string& name, const nvinfer1::Dims& dims) {
    if (!context_) return false;
    return context_->setInputShape(name.c_str(), dims);
}

bool TRTEngine::setTensorAddress(const std::string& name, void* devicePtr) {
    if (!context_) return false;
    return context_->setTensorAddress(name.c_str(), devicePtr);
}

bool TRTEngine::enqueueV3(cudaStream_t stream) {
    if (!context_) return false;
    return context_->enqueueV3(stream);
}

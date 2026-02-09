cmake_minimum_required(VERSION 3.20)
project(VibeVoiceServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── MSVC settings (CXX only, not CUDA) ──
add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)

# ── CUDA Toolkit ──
find_package(CUDAToolkit REQUIRED)

# ── TensorRT ──
find_path(TRT_INCLUDE NvInfer.h
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} PATH_SUFFIXES include)
find_library(TRT_LIB nvinfer
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} PATH_SUFFIXES lib)
find_library(TRT_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} PATH_SUFFIXES lib)
if(NOT TRT_INCLUDE OR NOT TRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT.")
endif()

# ── cuDNN ──
find_path(CUDNN_INCLUDE cudnn.h
    HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
find_library(CUDNN_LIB cudnn
    HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} PATH_SUFFIXES lib lib/x64)

# ── .cu files (optional) ──
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
if(CUDA_SOURCES)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "86;89;90")
endif()

# ── Sources ──
file(GLOB_RECURSE CXX_SOURCES src/*.cpp)
add_executable(vibevoice_server ${CXX_SOURCES} ${CUDA_SOURCES})

# MSVC warnings + parallel compile for C++ sources only
if(MSVC)
    target_compile_options(vibevoice_server PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /MP>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /Oi /Ot /GL /GS->
    )
    target_link_options(vibevoice_server PRIVATE
        $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>
    )
endif()

target_include_directories(vibevoice_server PRIVATE
    src
    third_party
    third_party/nlohmann
    third_party/cpp-httplib
    third_party/dr_libs
    third_party/stb
    ${TRT_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(CUDNN_INCLUDE)
    target_include_directories(vibevoice_server PRIVATE ${CUDNN_INCLUDE})
endif()

target_link_libraries(vibevoice_server PRIVATE
    CUDA::cudart
    CUDA::cublas
    ${TRT_LIB}
    ${TRT_PLUGIN}
    ws2_32
)

if(CUDNN_LIB)
    target_link_libraries(vibevoice_server PRIVATE ${CUDNN_LIB})
endif()

install(TARGETS vibevoice_server RUNTIME DESTINATION bin)

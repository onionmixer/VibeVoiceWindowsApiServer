@echo off
chcp 65001 >nul 2>&1
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set TENSORRT_ROOT=C:\Users\onion\Desktop\Workspace\TensorRT
set CUDNN_ROOT=C:\Users\onion\Desktop\Workspace\cudnn
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

cd /d "%~dp0"

echo === CMake Configure ===
cmake --preset release
if %ERRORLEVEL% neq 0 (
    echo CMake configure FAILED
    exit /b 1
)

echo === CMake Build ===
cmake --build --preset release
if %ERRORLEVEL% neq 0 (
    echo CMake build FAILED
    exit /b 1
)

echo === Build Successful ===

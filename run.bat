@echo off
chcp 65001 >nul 2>&1
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;C:\Users\onion\Desktop\Workspace\TensorRT\lib;C:\Users\onion\Desktop\Workspace\cudnn\bin;%PATH%
"%~dp0build\release\vibevoice_server.exe"

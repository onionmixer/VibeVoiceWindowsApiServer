# VibeVoice Windows Standalone API Server

## Project Summary
Docker 기반 Python VibeVoice API 서버를 Windows Native C++ Standalone 서비스로 포팅하는 프로젝트.
OpenAI-compatible REST API (TTS/STT) 를 제공한다.

## Reference Codebase
- `/mnt/USERS/onion/DATA_ORIGN/Workspace/VibeVoiceDockerApiServer/VibeVoiceDockerOpenaiApiServer/` — FastAPI REST API 서버 (Python)
- `/mnt/USERS/onion/DATA_ORIGN/Workspace/VibeVoiceDockerApiServer/VibeVoice/` — 핵심 ML 모델 라이브러리 (Python)
- `/mnt/USERS/onion/DATA_ORIGN/Workspace/VibeVoiceDockerApiServer/models/` — Pre-trained 모델 가중치 (safetensors) + 음성 프리셋

## Toolchain
- **Language**: C++17
- **Compiler**: clang-cl (LLVM/Clang MSVC 호환 드라이버)
- **Linker**: lld-link
- **Build System**: CMake
- **Target OS**: Windows

## Dependencies
- **Runtime (사용자 설치 필요)**: NVIDIA CUDA Toolkit >= 12.x, NVIDIA TensorRT >= 10.x
- **Build-time (소스 트리에 포함, 설치 불필요)**:
  - cpp-httplib (header-only, HTTP 서버)
  - nlohmann/json (header-only, JSON)
  - dr_wav, dr_mp3, dr_flac (header-only, audio I/O)
  - stb_vorbis (header-only, OGG decode)
- **CUDA library 외에 별도 설치가 필요한 외부 의존성은 없어야 한다**
- 모델 파일 외 필요한 파일(tokenizer, ffmpeg.exe 등)은 프로젝트 디렉터리에 포함

## Architecture
- TensorRT로 모델 추론 (safetensors → ONNX → TensorRT engine)
- 5개 서브모델 분리: acoustic_encoder, semantic_encoder, language_model, diffusion_head, acoustic_decoder
- BPE Tokenizer (Qwen2) C++ 직접 구현
- DPM-Solver (diffusion scheduler) C++ 구현
- HTTP API: cpp-httplib 기반
- Audio I/O: dr_libs (직접) + ffmpeg.exe subprocess (mp3/opus/aac/flac 변환)

## Key Constraints
1. C++로 제작
2. 모델: `/mnt/USERS/onion/DATA_ORIGN/Workspace/VibeVoiceDockerApiServer/models/` 사용
3. NVIDIA CUDA library 외 다른 것은 설치할 필요 없어야 함 (TensorRT는 NVIDIA 제품으로 허용)
4. 필요한 파일은 현재 디렉터리로 복사
5. clang-cl로 Windows 빌드

## Development Plan
상세 개발 계획은 `DEVPLAN.md` 참조.

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
- **Compiler**: MSVC (cl.exe)
- **Linker**: link.exe
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

## Precision Policy
- **기본 정밀도: FP32** — Python 레퍼런스 서버와 동일한 출력 품질을 보장하기 위해 FP32를 기본으로 사용
- ONNX export 및 TensorRT 엔진 빌드 모두 FP32 기본 (`build_engines.py --fp16` 명시 시에만 FP16 사용)
- FP16 Tensor Core(Ampere+)에서의 정밀도 손실로 인해 autoregressive feedback loop에서 누적 오차가 발생하므로, LM은 반드시 FP32로 운용
- Audio 컴포넌트(acoustic_encoder, acoustic_decoder, semantic_encoder, diffusion_head)는 FP16 허용 가능

## Architecture
- TensorRT로 모델 추론 (safetensors → ONNX → TensorRT engine)
- 0.5B TTS: 6 engines (base_lm_prefill/decode, tts_lm_prefill/decode, diffusion_head, acoustic_decoder) — split LM, binary voice presets
- 1.5B TTS: 8 engines (language_model_prefill/decode, acoustic_encoder, semantic_encoder, streaming_semantic_encoder, streaming_acoustic_decoder, diffusion_head, acoustic_decoder) — unified LM, WAV voice, streaming semantic feedback loop
- 1.5B LM: FP32 ONNX, FP32 KV-cache — Python 레퍼런스와 동일 정밀도
- 1.5B Autoregressive Loop: LM → diffusion → streaming acoustic decoder (with cache) → streaming semantic encoder (with cache) → connectors → LM
- BPE Tokenizer (Qwen2) C++ 직접 구현
- DPM-Solver++ (cosine schedule, v_prediction) C++ 구현
- GPU 연산: cuBLAS (matMul, vectorAdd, RMSNorm, embedding lookup)
- GPU 버퍼 관리: 루프 내 CudaBuffer는 클래스 멤버로 사전 할당하여 GPU 메모리 파편화 방지
- HTTP API: cpp-httplib 기반
- Audio I/O: dr_libs (직접) + ffmpeg.exe subprocess (mp3/opus/aac/flac 변환)

## Current Status
- **0.5B TTS**: 동작 완료 (스트리밍)
- **1.5B TTS**: 동작 완료 — Python 레퍼런스 대비 85~123% 범위의 유사한 오디오 길이, 멀티 리퀘스트 안정 (연속 10+ 요청 검증)
  - CFG=1.5, 20 diffusion steps, FP32 LM, streaming semantic/acoustic cache
- **STT/ASR**: 코드 구현 완료, 엔진 미빌드

## Key Constraints
1. C++로 제작
2. 모델: `/mnt/USERS/onion/DATA_ORIGN/Workspace/VibeVoiceDockerApiServer/models/` 사용
3. NVIDIA CUDA library 외 다른 것은 설치할 필요 없어야 함 (TensorRT는 NVIDIA 제품으로 허용)
4. 필요한 파일은 현재 디렉터리로 복사
5. MSVC (cl.exe)로 Windows 빌드

## Development Plan
상세 개발 계획은 `DEVPLAN.md` 참조.

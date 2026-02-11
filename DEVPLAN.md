# VibeVoice Windows Standalone API Server - Development Plan

## 1. Project Overview

### 1.1 Goal
Docker 기반 Python VibeVoice API 서버를 **Windows Native C++ Standalone 서비스**로 포팅.
NVIDIA CUDA/TensorRT만을 외부 의존성으로 사용하며, OpenAI-compatible REST API를 제공한다.

### 1.2 Toolchain
- **Compiler**: MSVC (cl.exe)
- **Build System**: CMake
- **GPU Inference**: NVIDIA TensorRT
- **GPU Runtime**: CUDA Toolkit (cuBLAS, cuDNN, cudart)

### 1.3 Reference Source
| Directory | Description |
|-----------|-------------|
| `VibeVoiceDockerOpenaiApiServer/` | FastAPI 기반 OpenAI-compatible REST API 서버 |
| `VibeVoice/vibevoice/` | 핵심 ML 모델 라이브러리 (tokenizer, diffusion, transformer) |
| `models/` | Pre-trained 모델 가중치 (safetensors) + 음성 프리셋 |

### 1.4 Supported Features
| Feature | Endpoint | Model | Params |
|---------|----------|-------|--------|
| TTS (Streaming) | `POST /v1/audio/speech` | VibeVoice-Realtime-0.5B | 0.5B |
| TTS (Full Quality) | `POST /v1/audio/speech` | VibeVoice-1.5B | 1.5B |
| STT (Transcription) | `POST /v1/audio/transcriptions` | VibeVoice-ASR | 7B |
| STT (Translation) | `POST /v1/audio/translations` | VibeVoice-ASR | 7B |

---

## 2. Architecture

### 2.1 High-Level

```
┌──────────────────────────────────────────────────────────────┐
│                  VibeVoice Windows API Server                 │
│                                                               │
│  ┌─────────────┐   ┌───────────────────────────────────────┐ │
│  │ HTTP Server  │   │          Inference Engine              │ │
│  │ (cpp-httplib)│   │                                       │ │
│  │             │   │  ┌───────────┐   ┌─────────────────┐  │ │
│  │ /v1/audio/  │───│  │TTS Pipeline│   │  STT Pipeline   │  │ │
│  │   speech    │   │  │           │   │                 │  │ │
│  │ /v1/audio/  │   │  │ TextToken │   │  AudioLoad      │  │ │
│  │ transcribe  │   │  │    ↓      │   │     ↓           │  │ │
│  │ /v1/audio/  │   │  │ LM Infer  │   │  Acoustic/Sem   │  │ │
│  │ translations│   │  │    ↓      │   │  Tokenize       │  │ │
│  │ /health     │   │  │ Diffusion │   │     ↓           │  │ │
│  │ /v1/models  │   │  │    ↓      │   │  LM Infer       │  │ │
│  │             │   │  │ AcDecode  │   │     ↓           │  │ │
│  │             │   │  │    ↓      │   │  TextDecode     │  │ │
│  │             │   │  │   WAV     │   │     ↓           │  │ │
│  └─────────────┘   │  └───────────┘   │  Formatted Out  │  │ │
│                     │                  └─────────────────┘  │ │
│                     │                                       │ │
│                     │  ┌─────────────────────────────────┐  │ │
│                     │  │       TensorRT Engines           │  │ │
│                     │  │  AcEnc SemEnc Qwen2 DiffHd AcDec│  │ │
│                     │  └─────────────────────────────────┘  │ │
│                     └───────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │BPE Tokenizer│  │ Audio I/O    │  │ Config Manager     │  │
│  │(Qwen2, C++) │  │ (dr_libs +   │  │ (nlohmann/json)    │  │
│  │             │  │  ffmpeg.exe) │  │                    │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Sub-Model Decomposition

원본 monolithic 모델을 TensorRT용 서브모델로 분리:

**0.5B (6 engines)**: base_lm_prefill, base_lm_decode, tts_lm_prefill, tts_lm_decode, diffusion_head, acoustic_decoder
**1.5B (6 engines)**: language_model_prefill, language_model_decode, acoustic_encoder, semantic_encoder, diffusion_head, acoustic_decoder

| Sub-Model | Input | Output | Description |
|-----------|-------|--------|-------------|
| `acoustic_encoder` | `[B,1,N]` audio | `[B,T,64]` latent | 24kHz→64d latent @7.5Hz (1.5B) |
| `semantic_encoder` | `[B,1,N]` audio | `[B,128,T]` semantic | 24kHz→128d semantic (1.5B, channel-first) |
| `language_model_prefill` | `[B,S,H]` embeds + mask | `[B,S,V]` logits + `[B,S,H]` hidden + KV | Qwen2 1.5B prefill (unified LM) |
| `language_model_decode` | `[B,1,H]` embed + mask + KV | `[B,1,V]` logits + `[B,1,H]` hidden + KV | Qwen2 1.5B decode (unified LM) |
| `base_lm_prefill/decode` | `[B,S]` tokens + KV-cache | `[B,S,H]` hidden + KV-cache | Qwen2 0.5B base LM (split) |
| `tts_lm_prefill/decode` | `[B,S]` tokens + KV-cache | `[B,S,H]` hidden + KV-cache | Qwen2 0.5B TTS LM (split) |
| `diffusion_head` | `[B,1,64]` noise + cond + t | `[B,1,64]` v_pred | Single denoise step |
| `acoustic_decoder` | `[B,64,T]` latent | `[B,1,T*3200]` audio | 64d latent→24kHz audio |

### 2.3 Inference Pipelines

**TTS 0.5B (Streaming)**:
```
Text → BPE → tokens
Voice .bin → KV-cache (GPU)
tokens + voice_kv → LM (autoregressive) → semantic tokens
For each chunk:
  noise → DPM loop(5 steps) { Diffusion Head } → denoised latent
  latent → Acoustic Decoder → audio chunk
Concat chunks → WAV
```

**TTS 1.5B (Full)**:
```
Text → BPE → tokens
Voice .wav → Acoustic Encoder → voice_latents [T, 64]
  → normalize: (latent + bias) * scale
  → Acoustic Connector: [T, 64] → [T, H] voice embeddings

Prompt = [sys_tokens + speech_start + voice_embeds + speech_end + text_tokens + speech_start]
Negative = [speech_start] (1 token for CFG)

Prefill → KV-cache + initial logits

Autoregressive loop:
  logits = constrain to {speech_diffusion, speech_end, eos}
  token = argmax(logits)
  if speech_diffusion:
    hidden_pos/neg → DPM loop(20 steps, CFG=3.0) → latent [64]
    denorm: latent / scale - bias
    decode latent → audio → Semantic Encoder → semantic [128]
    Acoustic Connector(latent) + Semantic Connector(semantic) → next embedding
  if speech_end/eos: stop

Batch decode all latents → Acoustic Decoder → WAV
```

**STT (ASR)**:
```
Audio → 24kHz mono float32
Audio → Acoustic Encoder → acoustic_tokens
Audio → Semantic Encoder → semantic_tokens
[prompt + acoustic + semantic] → LM (autoregressive) → text_tokens
text_tokens → BPE decode → text + segments + timestamps
```

---

## 3. Dependencies

### 3.1 Runtime (사용자 설치)

| Dependency | Version | Notes |
|------------|---------|-------|
| **CUDA Toolkit** | >= 12.x | cudart, cuBLAS, cuDNN |
| **TensorRT** | >= 10.x | NVIDIA inference runtime |

### 3.2 Build-time (소스 트리에 포함, 설치 불필요)

| Library | Type | Purpose | License |
|---------|------|---------|---------|
| **cpp-httplib** | Header-only | HTTP 서버 | MIT |
| **nlohmann/json** | Header-only | JSON | MIT |
| **dr_wav** | Header-only | WAV I/O | Public Domain |
| **dr_mp3** | Header-only | MP3 decode | Public Domain |
| **dr_flac** | Header-only | FLAC decode | Public Domain |
| **stb_vorbis** | Header-only | OGG decode | Public Domain |

### 3.3 프로젝트 파일 구조

```
VibeVoiceWindowsApiServer/
├── models/                          ← symlink or copy from Docker project
│   ├── VibeVoice-ASR/               (config.json + 8x safetensors ~17GB)
│   ├── VibeVoice-Realtime-0.5B/     (config.json + model.safetensors ~2GB)
│   ├── VibeVoice-1.5B/              (config.json + 3x safetensors ~5.4GB)
│   └── voices/
│       ├── streaming_model/         (.pt → .bin 변환)
│       └── full_model/              (.wav 그대로)
├── tokenizer/                       ← HuggingFace에서 다운로드
│   ├── qwen2.5-0.5b/tokenizer.json
│   ├── qwen2.5-1.5b/tokenizer.json
│   └── qwen2.5-7b/tokenizer.json
├── engines/                         ← TensorRT 엔진 (최초 1회 빌드)
│   ├── tts_0.5b/  (6 engines + weights)
│   ├── tts_1.5b/  (6 engines + weights)
│   └── asr/       (3 engines)
└── tools/
    └── ffmpeg.exe                   ← static build (mp3/opus/aac/flac 변환)
```

---

## 4. Model Conversion Pipeline

최초 1회 Python 스크립트로 실행. 런타임에는 불필요.

```
safetensors → [export_onnx.py] → ONNX → [build_engines.py / trtexec] → .trt
```

### 4.1 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/export_onnx.py` | 서브모델을 개별 ONNX로 export (0.5B: 6개, 1.5B: 6개) |
| `scripts/build_engines.py` | ONNX → TensorRT engine 빌드 |
| `scripts/convert_voices.py` | .pt (PyTorch KV-cache) → .bin (raw fp16) |
| `scripts/prepare_tokenizer.py` | HuggingFace tokenizer 다운로드 + special tokens 추가 |

### 4.2 ONNX Export 상세

```python
# acoustic_encoder: Conv1D stack (causal, depthwise)
#   Input:  audio [B, 1, N]  dynamic N
#   Output: latent_mean [B, T, 64]

# semantic_encoder: Conv1D stack
#   Input:  audio [B, 1, N]
#   Output: semantic_mean [B, 128, T]  (channel-first!)

# 1.5B language_model (unified Qwen2 with lm_head):
#   Prefill: inputs_embeds [B, S, H], attention_mask [B, S] fp32
#          → logits [B, S, V] fp32, hidden_states [B, S, H] fp16, kv_cache_out fp16
#   Decode:  inputs_embeds [B, 1, H], attention_mask [B, S+1] fp32, kv_cache_in fp16
#          → logits [B, 1, V] fp32, hidden_states [B, 1, H] fp16, kv_cache_out fp16

# 0.5B language_model (split base_lm + tts_lm, no lm_head):
#   Prefill: input_ids [B, S] → hidden_states [B, S, H] fp32, kv_cache_out fp32
#   Decode:  input_ids [B, 1] + kv_cache_in → hidden_states [B, 1, H] fp32, kv_cache_out fp32

# diffusion_head: single denoise step
#   Input:  noisy [B, 1, 64], cond [B, 1, H], timestep [B]
#   Output: v_prediction [B, 1, 64]

# acoustic_decoder: TransposedConv1D stack
#   Input:  latent [B, 64, T]  (channel-first)
#   Output: audio [B, 1, T*3200]
```

### 4.3 TensorRT Engine Build

```bash
trtexec --onnx=acoustic_encoder.onnx --saveEngine=acoustic_encoder.trt --fp16 \
        --minShapes=audio:1x1x24000 --optShapes=audio:1x1x480000 --maxShapes=audio:1x1x14400000
```

### 4.4 Voice Preset Conversion

```
.pt (PyTorch KV-cache) → .bin
Header: [u32 num_layers, u32 num_heads, u32 head_dim, u32 seq_len]
Body:   fp16 key[layers][heads][seq][dim] + value[layers][heads][seq][dim]
```

---

## 5. C++ Implementation

### 5.1 Source Layout

```
src/
├── main.cpp
├── server/
│   └── http_server.h/.cpp         # cpp-httplib HTTP 서버 + 모든 라우트
├── inference/
│   ├── trt_engine.h/.cpp          # TensorRT 엔진 로더/실행
│   ├── tts_pipeline.h/.cpp        # TTS 파이프라인 (0.5B 스트리밍 + 1.5B 풀)
│   ├── stt_pipeline.h/.cpp        # STT 파이프라인 오케스트레이션
│   ├── dpm_solver.h/.cpp          # DPM-Solver++ 스케줄러 (cosine, v_prediction)
│   ├── kv_cache.h/.cpp            # KV-cache GPU 메모리 관리 (fp16/fp32)
│   ├── gpu_ops.h                  # cuBLAS 기반 GPU 연산 (vectorAdd, matMul, RMSNorm 등)
│   └── model_config.h/.cpp        # 모델 메타데이터 + 가중치 파서
├── tokenizer/
│   ├── bpe_tokenizer.h/.cpp       # Qwen2 BPE (C++ 구현)
│   ├── unicode_utils.h/.cpp       # UTF-8 처리
│   └── special_tokens.h           # VibeVoice special token IDs
├── audio/
│   ├── audio_io.h/.cpp            # WAV/MP3/FLAC/OGG 읽기쓰기 (dr_libs)
│   ├── audio_convert.h/.cpp       # ffmpeg.exe subprocess 호출 (mp3/opus/aac/flac)
│   └── audio_libs_impl.cpp        # dr_libs 헤더 구현체
├── tests/
│   ├── test_common.h              # 테스트 공통 유틸리티
│   ├── tests_unit.cpp             # 유닛 테스트 (tokenizer, config 등)
│   └── tests_e2e.cpp              # E2E 테스트 (HTTP 엔드포인트)
└── utils/
    ├── cuda_buffer.h/.cpp         # CudaBuffer RAII 래퍼
    ├── cuda_check.h               # CUDA 에러 체크 매크로
    └── logger.h/.cpp              # 구조화 로깅 (타임스탬프, 레벨, 모듈, 요청ID)
```

### 5.2 Core Classes

**TRTEngine** — TensorRT 엔진 래퍼:
```cpp
class TRTEngine {
public:
    explicit TRTEngine(const std::string& engine_path, int device_id = 0);
    bool execute(const std::map<std::string, void*>& bindings,
                 const std::map<std::string, std::vector<int>>& shapes,
                 cudaStream_t stream);
    void setInputShape(const std::string& name, const std::vector<int>& shape);
private:
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    nvinfer1::IRuntime* runtime_;
};
```

**TTSPipeline** — TTS 오케스트레이션:
```cpp
class TTSPipeline {
public:
    enum class ModelType { STREAMING_0_5B, FULL_1_5B };
    struct Config { ModelType type; std::string engineDir, weightsDir, voicesDir, ...; float cfgScale; int inferenceSteps; };
    struct Request  { std::string text, voice; float speed; };
    struct Result   { std::vector<float> audio; int sampleRate; bool ok; std::string error; };

    bool load(const Config& cfg);
    Result synthesize(const Request& req);
    std::vector<std::string> availableVoices() const;
private:
    Result synth05B(const Request& req);   // 0.5B: split LM + binary voice presets
    Result synth15B(const Request& req);   // 1.5B: unified LM + WAV voice + semantic feedback
    // Helpers: sampleSpeechTokens, runConnector, runEosClassifier, runLmPrefill, runLmDecodeWithLogits, ...
    // 0.5B engines: baseLmPrefill_, baseLmDecode_, ttsLmPrefill_, ttsLmDecode_, diffusionHead_, acousticDecoder_
    // 1.5B engines: lmPrefill_, lmDecode_, acousticEncoder_, semanticEncoder_, diffusionHead_, acousticDecoder_
    // KV caches, GPU weights, scratch buffers, tokenizer, DPM solver, cuBLAS, ...
};
```

**STTPipeline** — STT 오케스트레이션:
```cpp
class STTPipeline {
public:
    struct Request { std::vector<float> audio; std::string lang, prompt; float temp; };
    struct Segment { float t0, t1; std::string text, speaker; };
    struct Result  { std::string text; std::vector<Segment> segs; bool ok; std::string err; };

    explicit STTPipeline(const ModelConfig& cfg);
    bool load();
    Result transcribe(const Request& req);
private:
    std::unique_ptr<TRTEngine> ac_enc_, sem_enc_, lm_;
    std::unique_ptr<BPETokenizer> tok_;
};
```

**BPETokenizer** — Qwen2 BPE:
```cpp
class BPETokenizer {
public:
    bool load(const std::string& tokenizer_json_path);
    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    int32_t speech_start_id, speech_end_id, speech_pad_id, eos_id;
};
```

**DPMSolver** — Diffusion 스케줄러:
```cpp
class DPMSolver {
public:
    DPMSolver(int train_steps, int infer_steps, const std::string& schedule, const std::string& pred_type);
    std::vector<int> timesteps() const;
    void step(float* sample, const float* v_pred, int t, int t_prev, cudaStream_t s);
private:
    std::vector<float> alphas_cumprod_;
};
```

**HttpServer** — API 서버:
```cpp
class HttpServer {
public:
    HttpServer(const ServerConfig& cfg, TTSPipeline* tts05, TTSPipeline* tts15, STTPipeline* stt);
    void start();  // blocking
    void stop();
};
```

### 5.3 API Endpoints

| Method | Path | Content-Type | Description |
|--------|------|-------------|-------------|
| POST | `/v1/audio/speech` | application/json | TTS → binary audio |
| POST | `/v1/audio/transcriptions` | multipart/form-data | STT → json/text/srt/vtt |
| POST | `/v1/audio/translations` | multipart/form-data | STT(→EN) |
| GET | `/health` | — | 서비스 상태 |
| GET | `/v1/models` | — | 사용 가능 모델 |
| GET | `/v1/audio/voices` | — | 사용 가능 보이스 |

### 5.4 Runtime Config (`config.json`)

```json
{
    "server": { "host": "0.0.0.0", "port": 8080 },
    "models": {
        "tts_0.5b": {
            "enabled": true,
            "engine_dir": "engines/tts_0.5b",
            "weights_dir": "engines/tts_0.5b/weights",
            "metadata_path": "engines/tts_0.5b/model_metadata.json",
            "voices_dir": "voices/streaming_model",
            "tokenizer_path": "tokenizer/qwen2.5-0.5b/tokenizer.json",
            "special_tokens_path": "tokenizer/qwen2.5-0.5b/special_tokens.json",
            "cfg_scale": 1.5, "inference_steps": 20
        },
        "tts_1.5b": {
            "enabled": true,
            "engine_dir": "engines/tts_1.5b",
            "weights_dir": "engines/tts_1.5b/weights",
            "metadata_path": "engines/tts_1.5b/model_metadata.json",
            "voices_dir": "voices/full_model",
            "tokenizer_path": "tokenizer/qwen2.5-1.5b/tokenizer.json",
            "special_tokens_path": "tokenizer/qwen2.5-1.5b/special_tokens.json",
            "cfg_scale": 3.0, "inference_steps": 20
        },
        "asr": {
            "enabled": false,
            "engine_dir": "engines/asr",
            "max_new_tokens": 32768, "temperature": 0.0
        }
    },
    "defaults": {
        "tts_voice": "en-Carter_man",
        "tts_model": "tts_0.5b",
        "max_audio_duration": 600
    }
}
```

---

## 6. Build System (MSVC + CMake)

### 6.1 Toolchain

| Tool | Version | Role |
|------|---------|------|
| **MSVC (cl.exe)** | VS 2019+ | C++ 컴파일 |
| **CMake** | >= 3.20 | 빌드 설정 + 빌드 백엔드 |
| **nvcc** | CUDA 12.x | .cu 파일 컴파일 (필요 시) |

MSVC는 TensorRT/CUDA 라이브러리(.lib)와 네이티브로 링크 가능하다.
`.cu` 파일이 필요한 경우 nvcc로 분리 컴파일 후 링크한다.

### 6.2 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(VibeVoiceServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── MSVC 설정 ──
if(MSVC)
    add_compile_options(/W4 /MP)
endif()
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

# ── .cu files (optional) ──
file(GLOB CUDA_SOURCES src/**/*.cu)
if(CUDA_SOURCES)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "86;89;90")
endif()

# ── Sources ──
file(GLOB_RECURSE CXX_SOURCES src/*.cpp)
add_executable(vibevoice_server ${CXX_SOURCES} ${CUDA_SOURCES})

target_include_directories(vibevoice_server PRIVATE
    src  third_party  third_party/nlohmann  third_party/cpp-httplib  third_party/dr_libs
    ${TRT_INCLUDE}  ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(vibevoice_server PRIVATE
    CUDA::cudart  CUDA::cublas  ${TRT_LIB}  ${TRT_PLUGIN}  ws2_32)

install(TARGETS vibevoice_server RUNTIME DESTINATION bin)
```

### 6.3 CMakePresets.json

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "release",
            "displayName": "MSVC Release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build/release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "TENSORRT_ROOT": "$env{TENSORRT_ROOT}",
                "CUDNN_ROOT": "$env{CUDNN_ROOT}",
                "CUDAToolkit_ROOT": "$env{CUDA_PATH}"
            }
        },
        {
            "name": "debug",
            "displayName": "MSVC Debug",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build/debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "TENSORRT_ROOT": "$env{TENSORRT_ROOT}",
                "CUDNN_ROOT": "$env{CUDNN_ROOT}",
                "CUDAToolkit_ROOT": "$env{CUDA_PATH}"
            }
        }
    ],
    "buildPresets": [
        { "name": "release", "configurePreset": "release" },
        { "name": "debug",   "configurePreset": "debug" }
    ]
}
```

### 6.4 Build Commands

```batch
:: 환경: VS Developer Command Prompt (MSVC CRT headers/libs 경로 확보)
:: 또는 vcvarsall.bat x64 실행 후 진행

set TENSORRT_ROOT=C:\TensorRT-10.x

:: 모델 변환 (최초 1회)
python scripts\prepare_tokenizer.py
python scripts\export_onnx.py --model-dir models --output-dir onnx
python scripts\build_engines.py --onnx-dir onnx --output-dir engines
python scripts\convert_voices.py --voices-dir models\voices

:: 빌드
cmake --preset release
cmake --build --preset release

:: 실행
build\release\vibevoice_server.exe --config config.json
```

### 6.5 MSVC 호환성

| 대상 | 호환성 |
|------|--------|
| TensorRT C++ API (.lib) | 네이티브 링크 |
| CUDA Runtime (.lib) | 네이티브 링크 |
| cuBLAS/cuDNN (.lib) | 네이티브 링크 |
| .cu device code | nvcc가 MSVC host compiler 사용 |
| Windows API (ws2_32 등) | 완전 호환 |

---

## 7. Development Phases

### Phase 0: 프로젝트 셋업 (1-2일)
- [x] CMakeLists.txt + CMakePresets.json
- [x] third_party/ 다운로드 (cpp-httplib, nlohmann/json, dr_libs)
- [x] 디렉터리 구조 생성
- [x] MSVC + TensorRT + CUDA 링크 확인 (빌드 테스트)
- [x] main.cpp 스켈레톤

### Phase 1: Model Conversion Pipeline (3-5일)
- [x] `export_onnx.py` — 서브모델 ONNX export
  - [x] Acoustic encoder (0.5B, 1.5B)
  - [x] Semantic encoder (1.5B)
  - [x] Language model with KV-cache (0.5B: split base_lm/tts_lm, 1.5B: unified with lm_head)
  - [x] Diffusion head
  - [x] Acoustic decoder
  - [x] 1.5B FP16 ONNX export (inputs_embeds/hidden_states fp16, logits/mask fp32)
- [x] `build_engines.py` — ONNX → TensorRT
- [x] `convert_voices.py` — .pt → .bin (0.5B streaming voices)
- [x] `prepare_tokenizer.py` — Qwen2 tokenizer 준비
- [x] ONNX output vs 원본 output 비교 검증 (0.5B, 1.5B 모두 동작 확인)

### Phase 2: Core C++ (3-4일)
- [x] TRTEngine — 엔진 로더/실행
- [x] CudaBuffer — GPU 메모리 관리
- [x] BPETokenizer — Qwen2 BPE C++ 구현
- [x] ModelConfig — config.json 파서
- [x] AudioIO — WAV/MP3/FLAC/OGG 로드 + WAV 쓰기
- [x] AudioConvert — ffmpeg.exe subprocess

### Phase 3: TTS Pipeline (5-7일)
- [x] 0.5B Streaming
  - [x] Voice .bin 로드 → GPU (KV-cache presets)
  - [x] Text → BPE → tokens
  - [x] Split LM autoregressive (base_lm + tts_lm, 4 KV-caches for CFG)
  - [x] DPMSolver++ (cosine, v_prediction, configurable steps)
  - [x] Diffusion loop → Acoustic decode → WAV
  - [x] EOS classifier → speech end detection
- [x] 1.5B Full
  - [x] Voice .wav → Acoustic Encoder → latent → Acoustic Connector → voice embeddings
  - [x] Prompt construction (system + voice + text + special tokens)
  - [x] Unified LM prefill/decode with logits + hidden_states output
  - [x] Token-constrained generation (speech_diffusion, speech_end, eos)
  - [x] CFG: positive + negative path with KV-cache
  - [x] DPM-Solver++ diffusion (20 steps, CFG scale=3.0)
  - [x] Semantic feedback: decode → Semantic Encoder → Semantic Connector → next embedding
  - [x] Batch acoustic decode → WAV
  - [ ] Multi-speaker script parsing

### Phase 4: STT Pipeline (4-5일)
- [x] Audio 전처리 (resample, normalize, mono)
- [x] Acoustic/Semantic encoding
- [x] Prompt token 구성
- [x] LM autoregressive text generation
- [x] Text decode + segment/timestamp/speaker 파싱
- [x] SRT/VTT output

### Phase 5: HTTP API (2-3일)
- [x] POST /v1/audio/speech
- [x] POST /v1/audio/transcriptions
- [x] POST /v1/audio/translations
- [x] GET /health, /v1/models, /v1/audio/voices
- [x] Multipart form-data 파싱
- [x] Audio format conversion
- [x] CORS + error handling

### Phase 6: Integration & Optimization (3-4일)
- [x] Logger 시스템 (타임스탬프, 로그 레벨, 모듈 태그, 요청 ID, RAII 타이머)
- [x] fprintf → Logger 일괄 교체 (11개 소스 파일)
- [x] GPU 메모리 최적화 (per-request 임시 버퍼 → 멤버 scratch 버퍼 승격)
- [x] Thread safety (per-pipeline timed_mutex, graceful shutdown)
- [x] 테스트 리팩터링 (main.cpp 931줄 → 85줄, tests/ 분리)
- [x] E2E 테스트 (HTTP 엔드포인트 + 조건부 TRT 추론)

### Phase 7: 패키징 (1-2일)
- [x] Release 빌드 (/O2, LTCG)
- [x] 배포 패키지 (exe + engines + tokenizer + voices + ffmpeg)
- [x] 실행 가이드

---

## 8. Technical Challenges & Mitigations

| Challenge | Mitigation / Resolution |
|-----------|------------|
| **KV-cache in TensorRT** | Prefill/Decode 별도 프로파일, external GPU buffer, dynamic shapes. FP16 KV-cache로 메모리 절감 (1.5B: 224MB/cache) |
| **Custom ops (RoPE, RMSNorm)** | ONNX 기본 연산으로 분해. Connector/RMSNorm은 cuBLAS로 C++ 구현 |
| **BPE 정확도** | tokenizer.json 직접 파싱, Python output 비교 테스트 완료 |
| **FP16 수치 안정성** | 1.5B: ONNX를 FP16으로 export (embeds/hidden fp16, logits/mask fp32). 0.5B: FP32 엔진 사용 |
| **TRT 비동기 실행 경합** | neg/pos decode 간 cudaStreamSynchronize 삽입, 공유 버퍼를 persistent member로 승격 |
| **Semantic encoder 출력 레이아웃** | Channel-first [B,128,T] — stride 접근으로 프레임 추출 |
| **ASR 7B 메모리 (~16GB)** | 모듈별 선택 로딩, INT8 양자화 옵션 |
| **MSVC + nvcc 공존** | .cu는 nvcc로 컴파일 (MSVC를 host compiler로 사용) |

---

## 9. Audio Specs

| Parameter | Value |
|-----------|-------|
| Sample Rate | 24,000 Hz |
| Channels | Mono |
| Bit Depth (output) | 16-bit int |
| Internal | float32 [-1.0, 1.0] |
| Token Rate | 7.5 Hz (÷3200) |
| dB Normalize | -25 dB FS |

**Output**: wav(직접), pcm(직접), mp3/opus/aac/flac(ffmpeg.exe)
**Input(STT)**: wav/mp3/flac(dr_libs 직접), ogg(stb_vorbis), 기타(ffmpeg→wav)

---

## 10. Voice Mapping

**0.5B**: carter, davis, emma, frank, grace, mike, samuel (+ 다국어 18종) — `.bin` binary KV-cache preset
**1.5B**: 9개 WAV 파일 (예: en-Carter_man, en-Maya_woman 등) — `.wav` 참조 음성

**OpenAI aliases**: alloy→carter, echo→frank, fable→emma/alice, onyx→davis/carter, nova→grace/maya, shimmer→emma/alice

---

## 11. Performance Targets

| Metric | Target |
|--------|--------|
| TTS 0.5B | < 3s / 100 chars (RTX 3090+) |
| TTS 1.5B | < 8s / 100 chars |
| STT | < 1x realtime |
| Cold Start | < 30s |
| VRAM (TTS only) | < 6 GB |
| VRAM (All) | < 22 GB |

# VibeVoice Windows API Server

Windows Native C++ standalone server providing **OpenAI-compatible TTS/STT REST API** powered by NVIDIA TensorRT.

## Supported Models

| Model | API Name | Parameters | Use Case | VRAM |
|-------|----------|-----------|----------|------|
| VibeVoice-Realtime-0.5B | `vibevoice-0.5b` | 0.5B | Low-latency streaming TTS | ~2 GB |
| VibeVoice-1.5B | `vibevoice-1.5b` | 1.5B | High-quality TTS | ~4 GB |
| VibeVoice-ASR | `whisper-1` | 7B | Speech recognition / translation | ~16 GB |

---

## 1. System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| GPU | NVIDIA GPU (Compute Capability 7.0+) | RTX 3080 or better |
| VRAM | 4 GB (1.5B TTS only) | 8 GB+ (TTS 0.5B + 1.5B) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB (engines + models) | 50 GB+ |

### Software

| Component | Version | Purpose |
|-----------|---------|---------|
| **NVIDIA Driver** | 535+ | GPU driver |
| **CUDA Toolkit** | 12.x | GPU runtime (cudart, cuBLAS) |
| **TensorRT** | 8.6+ | Model inference engine |
| **cuDNN** | 8.x | Deep learning primitives |
| **Python** | 3.10+ | Model conversion (one-time setup) |
| **Visual Studio 2022 Build Tools** | v14.x | C++ compiler (building from source only) |
| **CMake** | 3.20+ | Build system (building from source only) |

---

## 2. Environment Setup

### 2.1 NVIDIA Driver

Install the latest NVIDIA Game Ready or Studio driver from https://www.nvidia.com/drivers.
Verify with:
```batch
nvidia-smi
```

### 2.2 CUDA Toolkit

1. Download CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads
2. Run the installer (default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)
3. The installer automatically adds CUDA to `PATH`. Verify:

```batch
nvcc --version
```

### 2.3 TensorRT

1. Download TensorRT 8.6+ (or 10.x) from https://developer.nvidia.com/tensorrt
   - Choose the **Windows zip package** matching your CUDA version
2. Extract to a directory, e.g. `C:\TensorRT`
3. Set environment variable:

```batch
set TENSORRT_ROOT=C:\TensorRT
```

4. Add TensorRT to PATH:

```batch
set PATH=%TENSORRT_ROOT%\lib;%TENSORRT_ROOT%\bin;%PATH%
```

5. Verify `trtexec` is available:

```batch
trtexec --help
```

### 2.4 cuDNN

1. Download cuDNN 8.x from https://developer.nvidia.com/cudnn (requires NVIDIA account)
   - Choose the **Windows zip** matching your CUDA version
2. Extract to a directory, e.g. `C:\cudnn`
3. Set environment variable:

```batch
set CUDNN_ROOT=C:\cudnn
```

4. Add cuDNN to PATH:

```batch
set PATH=%CUDNN_ROOT%\bin;%PATH%
```

### 2.5 Python (Model Conversion Only)

Python is required only for the one-time model conversion step.

1. Install Python 3.10+ from https://www.python.org/downloads/
2. Install dependencies:

```batch
pip install -r scripts\requirements.txt
```

This installs: `torch`, `transformers`, `safetensors`, `onnx`, `onnxruntime`, `numpy`, `diffusers`

### 2.6 VibeVoice Python Library (Model Conversion Only)

The ONNX export scripts require the VibeVoice Python library (contains model architecture definitions).

Set the path via environment variable:

```batch
set VIBEVOICE_LIB=C:\path\to\VibeVoice
```

Or pass it directly to the export script:

```batch
python scripts\export_onnx.py --vibevoice-lib C:\path\to\VibeVoice ...
```

---

## 3. Model Download

Download the pre-trained model weights (safetensors format):

| Model | Directory | Size | Content |
|-------|-----------|------|---------|
| VibeVoice-Realtime-0.5B | `VibeVoice-Realtime-0.5B/` | ~2 GB | `config.json` + `model.safetensors` |
| VibeVoice-1.5B | `VibeVoice-1.5B/` | ~5.4 GB | `config.json` + 3x `model-*.safetensors` |
| VibeVoice-ASR | `VibeVoice-ASR/` | ~17 GB | `config.json` + 8x `model-*.safetensors` |

Additionally, download the **voice presets**:

| Voice Type | Directory | Format | Description |
|------------|-----------|--------|-------------|
| 0.5B Streaming | `voices/streaming_model/` | `.pt` (PyTorch) | Pre-computed KV-cache voice presets |
| 1.5B Full | `voices/full_model/` | `.wav` (audio) | Reference voice audio files |

Place all downloads in a common directory, e.g. `C:\models`:
```
C:\models\
├── VibeVoice-Realtime-0.5B\    config.json + safetensors
├── VibeVoice-1.5B\              config.json + safetensors
├── VibeVoice-ASR\               config.json + safetensors
└── voices\
    ├── streaming_model\          *.pt files
    └── full_model\               *.wav files
```

---

## 4. Model Conversion

Model conversion is a **one-time process** that transforms safetensors weights into optimized TensorRT engines.

**Pipeline**: `safetensors` → `ONNX` → `TensorRT engine (.trt)`

### 4.1 Prepare Tokenizer

Downloads Qwen2.5 BPE tokenizer files from HuggingFace and generates VibeVoice special token mappings.

```batch
python scripts\prepare_tokenizer.py --output-dir tokenizer
```

Creates:
```
tokenizer\
├── qwen2.5-0.5b\    tokenizer.json + special_tokens.json
├── qwen2.5-1.5b\    tokenizer.json + special_tokens.json
└── qwen2.5-7b\      tokenizer.json + special_tokens.json
```

### 4.2 Export ONNX Models

Exports sub-models from safetensors to ONNX format. Also extracts connector weights as binary files.

**1.5B TTS** (requires CUDA GPU for FP16 export):
```batch
python scripts\export_onnx.py ^
    --model-dir C:\models\VibeVoice-1.5B ^
    --model-type tts_1.5b ^
    --output-dir onnx\tts_1.5b
```

**0.5B Streaming TTS**:
```batch
python scripts\export_onnx.py ^
    --model-dir C:\models\VibeVoice-Realtime-0.5B ^
    --model-type tts_0.5b ^
    --output-dir onnx\tts_0.5b
```

**ASR** (optional):
```batch
python scripts\export_onnx.py ^
    --model-dir C:\models\VibeVoice-ASR ^
    --model-type asr ^
    --output-dir onnx\asr
```

**1.5B export output:**
```
onnx\tts_1.5b\
├── acoustic_encoder.onnx
├── acoustic_decoder.onnx
├── semantic_encoder.onnx
├── language_model_prefill.onnx
├── language_model_decode.onnx
├── diffusion_head.onnx
├── model_metadata.json
└── weights\
    ├── embed_tokens.bin
    ├── acoustic_connector.bin
    ├── semantic_connector.bin
    ├── speech_scaling_factor.bin
    └── speech_bias_factor.bin
```

**0.5B export output** (split LM architecture — no acoustic/semantic encoder):
```
onnx\tts_0.5b\
├── base_lm_prefill.onnx
├── base_lm_decode.onnx
├── tts_lm_prefill.onnx
├── tts_lm_decode.onnx
├── acoustic_decoder.onnx
├── diffusion_head.onnx
├── model_metadata.json
└── weights\
    ├── embed_tokens.bin
    ├── tts_embed_tokens.bin
    ├── acoustic_connector.bin
    ├── tts_input_types.bin
    ├── tts_eos_classifier.bin
    ├── speech_scaling_factor.bin
    └── speech_bias_factor.bin
```

### 4.3 Build TensorRT Engines

Converts ONNX models to optimized TensorRT engines with dynamic shape profiles. Requires `trtexec` (included in TensorRT).

```batch
:: Ensure TENSORRT_ROOT is set so trtexec can be found
set TENSORRT_ROOT=C:\TensorRT

python scripts\build_engines.py ^
    --onnx-dir onnx\tts_1.5b ^
    --output-dir engines\tts_1.5b ^
    --fp16

python scripts\build_engines.py ^
    --onnx-dir onnx\tts_0.5b ^
    --output-dir engines\tts_0.5b ^
    --fp16
```

This step takes **10-30 minutes per model** depending on GPU. The script also copies `weights/` and `model_metadata.json` to the engine output directory.

Engine build options:
- `--fp16` (default: enabled) — FP16 precision for faster inference
- `--no-fp16` — Force FP32 precision
- `--only <name>` — Build only a specific engine (e.g. `--only acoustic_encoder`)

### 4.4 Convert Voice Presets

Converts voice presets to the format expected by the C++ server.

```batch
python scripts\convert_voices.py ^
    --voices-dir C:\models\voices ^
    --output-dir voices
```

- **0.5B streaming voices** (`.pt` → `.bin`): Converts PyTorch KV-cache presets to raw binary
- **1.5B full voices** (`.wav` → `.wav`): Copies WAV reference audio files as-is

Creates:
```
voices\
├── streaming_model\    *.bin files (one per voice)
└── full_model\         *.wav files (one per voice)
```

---

## 5. Configuration

Edit `config.json` to set server options and model paths.

### Full config.json Example

```json
{
    "server": {
        "host": "0.0.0.0",
        "port": 8080
    },
    "models": {
        "tts_0.5b": {
            "enabled": true,
            "engine_dir": "engines/tts_0.5b",
            "weights_dir": "engines/tts_0.5b/weights",
            "metadata_path": "engines/tts_0.5b/model_metadata.json",
            "voices_dir": "voices/streaming_model",
            "tokenizer_path": "tokenizer/qwen2.5-0.5b/tokenizer.json",
            "special_tokens_path": "tokenizer/qwen2.5-0.5b/special_tokens.json",
            "cfg_scale": 1.5,
            "inference_steps": 20
        },
        "tts_1.5b": {
            "enabled": true,
            "engine_dir": "engines/tts_1.5b",
            "weights_dir": "engines/tts_1.5b/weights",
            "metadata_path": "engines/tts_1.5b/model_metadata.json",
            "voices_dir": "voices/full_model",
            "tokenizer_path": "tokenizer/qwen2.5-1.5b/tokenizer.json",
            "special_tokens_path": "tokenizer/qwen2.5-1.5b/special_tokens.json",
            "cfg_scale": 3.0,
            "inference_steps": 20
        },
        "asr": {
            "enabled": false,
            "engine_dir": "engines/asr",
            "weights_dir": "engines/asr",
            "metadata_path": "engines/asr/metadata.json",
            "tokenizer_path": "tokenizer/qwen2.5-7b/tokenizer.json",
            "special_tokens_path": "tokenizer/qwen2.5-7b/special_tokens.json",
            "max_new_tokens": 32768,
            "temperature": 0.0
        }
    },
    "defaults": {
        "tts_voice": "en-Carter_man",
        "tts_model": "tts_0.5b",
        "max_audio_duration": 600
    }
}
```

### Configuration Fields

| Field | Description |
|-------|-------------|
| `server.host` | Listen address (`0.0.0.0` for all interfaces) |
| `server.port` | Listen port (default: 8080) |
| `models.*.enabled` | Set `true` only for models with built TensorRT engines |
| `models.*.engine_dir` | Path to directory containing `.trt` engine files |
| `models.*.weights_dir` | Path to directory containing `.bin` weight files |
| `models.*.metadata_path` | Path to `model_metadata.json` |
| `models.*.voices_dir` | Path to voice preset directory |
| `models.*.tokenizer_path` | Path to `tokenizer.json` |
| `models.*.special_tokens_path` | Path to `special_tokens.json` |
| `models.tts_*.cfg_scale` | Classifier-free guidance scale (0.5B: 1.5, 1.5B: 3.0) |
| `models.tts_*.inference_steps` | DPM-Solver++ diffusion steps (5-20, higher = better quality) |
| `defaults.tts_voice` | Default voice when not specified in request |

All paths are relative to the server executable's working directory, or can be absolute.

---

## 6. Running the Server

### Runtime PATH Setup

Before running the server, ensure TensorRT and cuDNN DLLs are in PATH:

```batch
set PATH=C:\TensorRT\lib;C:\TensorRT\bin;C:\cudnn\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin;%PATH%
```

### Start

```batch
vibevoice_server.exe --config config.json
```

The server will:
1. Print CUDA/GPU info
2. Load enabled TensorRT engines (takes ~10-15 seconds)
3. Start listening on the configured host:port

### CLI Options

```
vibevoice_server.exe [OPTIONS]

Options:
  --config <path>    Path to config.json (default: config.json)
  --test             Run unit tests and exit
  --test-e2e         Run end-to-end tests and exit
```

### Quick Test

```batch
curl -X POST http://localhost:8080/v1/audio/speech ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"vibevoice-1.5b\",\"input\":\"Hello, this is a test.\",\"voice\":\"en-Carter_man\",\"response_format\":\"wav\"}" ^
  --output test.wav
```

---

## 7. API Endpoints

### TTS — Text to Speech

```
POST /v1/audio/speech
Content-Type: application/json
```

Request body:
```json
{
    "model": "vibevoice-1.5b",
    "input": "Hello, world!",
    "voice": "en-Carter_man",
    "response_format": "wav",
    "speed": 1.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | `vibevoice-0.5b` (streaming) or `vibevoice-1.5b` (full quality) |
| `input` | string | required | Text to synthesize |
| `voice` | string | `"alloy"` | Voice name (see voices endpoint). OpenAI aliases supported. |
| `response_format` | string | `"wav"` | `wav`, `pcm`, `mp3`, `opus`, `aac`, `flac` |
| `speed` | float | `1.0` | Speech speed multiplier |

Response: binary audio in the requested format.

**OpenAI voice aliases**: alloy, echo, fable, onyx, nova, shimmer are mapped to native voices.

**Note**: `mp3`, `opus`, `aac`, `flac` formats require `ffmpeg.exe` in the `tools/` directory.

### STT — Speech to Text

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, flac, ogg, etc.) |
| `model` | string | `"whisper-1"` | Model identifier |
| `language` | string | auto | ISO 639-1 language code |
| `prompt` | string | `""` | Optional context hint |
| `response_format` | string | `"json"` | `json`, `text`, `srt`, `vtt`, `verbose_json` |
| `temperature` | float | `0.0` | Sampling temperature |

### STT — Translation

```
POST /v1/audio/translations
Content-Type: multipart/form-data
```

Same parameters as transcriptions. Translates audio to English.

### Health & Info

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health status |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/v1/audio/voices` | List available voices |

---

## 8. Building from Source

### 8.1 Prerequisites

| Tool | Download |
|------|----------|
| Visual Studio 2022 Build Tools | https://visualstudio.microsoft.com/downloads/ (select "C++ build tools" workload) |
| CMake 3.20+ | https://cmake.org/download/ |
| Ninja (optional) | https://github.com/nickvision-apps/ninja-builds |
| CUDA Toolkit 12.x | See Section 2.2 |
| TensorRT 8.6+ | See Section 2.3 |
| cuDNN 8.x | See Section 2.4 |

### 8.2 Build Steps

```batch
:: 1. Open VS Developer Command Prompt or initialize MSVC environment
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

:: 2. Set library paths
set TENSORRT_ROOT=C:\TensorRT
set CUDNN_ROOT=C:\cudnn

:: 3. Configure (generates build files)
cmake --preset release

:: 4. Build
cmake --build --preset release
```

The executable is produced at `build\release\vibevoice_server.exe`.

### 8.3 Build Presets

| Preset | Description |
|--------|-------------|
| `release` | Optimized build with /O2, LTCG |
| `debug` | Debug build with symbols |

---

## 9. Directory Structure

Complete directory layout after all setup steps:

```
VibeVoiceServer\
├── vibevoice_server.exe          Server executable
├── config.json                   Server configuration
│
├── engines\                      TensorRT engines (built from ONNX)
│   ├── tts_0.5b\
│   │   ├── base_lm_prefill.trt
│   │   ├── base_lm_decode.trt
│   │   ├── tts_lm_prefill.trt
│   │   ├── tts_lm_decode.trt
│   │   ├── diffusion_head.trt
│   │   ├── acoustic_decoder.trt
│   │   ├── model_metadata.json
│   │   └── weights\              embed_tokens.bin, acoustic_connector.bin, ...
│   └── tts_1.5b\
│       ├── language_model_prefill.trt
│       ├── language_model_decode.trt
│       ├── acoustic_encoder.trt
│       ├── semantic_encoder.trt
│       ├── diffusion_head.trt
│       ├── acoustic_decoder.trt
│       ├── model_metadata.json
│       └── weights\              embed_tokens.bin, acoustic_connector.bin, semantic_connector.bin, ...
│
├── tokenizer\                    BPE tokenizer files (from prepare_tokenizer.py)
│   ├── qwen2.5-0.5b\            tokenizer.json + special_tokens.json
│   ├── qwen2.5-1.5b\            tokenizer.json + special_tokens.json
│   └── qwen2.5-7b\              tokenizer.json + special_tokens.json
│
├── voices\                       Voice presets
│   ├── streaming_model\          0.5B: *.bin (converted from .pt)
│   └── full_model\               1.5B: *.wav (reference audio)
│
├── tools\
│   └── ffmpeg.exe                Required for mp3/opus/aac/flac output
│
├── onnx\                         ONNX models (intermediate, can be deleted after engine build)
│   ├── tts_0.5b\
│   └── tts_1.5b\
│
└── scripts\                      Python conversion scripts (not needed at runtime)
    ├── export_onnx.py
    ├── build_engines.py
    ├── convert_voices.py
    ├── prepare_tokenizer.py
    └── requirements.txt
```

---

## 10. Quick Start Summary

### 1.5B TTS (Full Quality) — End-to-end setup

```batch
:: Environment
set TENSORRT_ROOT=C:\TensorRT
set CUDNN_ROOT=C:\cudnn
set VIBEVOICE_LIB=C:\path\to\VibeVoice
set PATH=%TENSORRT_ROOT%\lib;%TENSORRT_ROOT%\bin;%CUDNN_ROOT%\bin;%PATH%

:: Python dependencies (one-time)
pip install -r scripts\requirements.txt

:: Step 1: Tokenizer
python scripts\prepare_tokenizer.py --output-dir tokenizer

:: Step 2: ONNX export (requires CUDA GPU for FP16 export)
python scripts\export_onnx.py --model-dir C:\models\VibeVoice-1.5B --model-type tts_1.5b --output-dir onnx\tts_1.5b

:: Step 3: TensorRT engine build (10-30 min)
python scripts\build_engines.py --onnx-dir onnx\tts_1.5b --output-dir engines\tts_1.5b --fp16

:: Step 4: Voice presets
python scripts\convert_voices.py --voices-dir C:\models\voices --output-dir voices

:: Step 5: Enable tts_1.5b in config.json, then run
vibevoice_server.exe --config config.json
```

### 0.5B TTS (Streaming) — End-to-end setup

```batch
:: Environment (same as above)
set TENSORRT_ROOT=C:\TensorRT
set CUDNN_ROOT=C:\cudnn
set VIBEVOICE_LIB=C:\path\to\VibeVoice
set PATH=%TENSORRT_ROOT%\lib;%TENSORRT_ROOT%\bin;%CUDNN_ROOT%\bin;%PATH%

:: Python dependencies (one-time, skip if already installed)
pip install -r scripts\requirements.txt

:: Step 1: Tokenizer (skip if already done)
python scripts\prepare_tokenizer.py --output-dir tokenizer

:: Step 2: ONNX export
python scripts\export_onnx.py --model-dir C:\models\VibeVoice-Realtime-0.5B --model-type tts_0.5b --output-dir onnx\tts_0.5b

:: Step 3: TensorRT engine build (10-30 min)
python scripts\build_engines.py --onnx-dir onnx\tts_0.5b --output-dir engines\tts_0.5b --fp16

:: Step 4: Voice presets (skip if already done — same command covers both models)
python scripts\convert_voices.py --voices-dir C:\models\voices --output-dir voices

:: Step 5: Enable tts_0.5b in config.json, then run
vibevoice_server.exe --config config.json
```

### Both Models

두 모델을 모두 사용하려면 Step 2-3을 각각 실행한 뒤, `config.json`에서 `tts_0.5b`와 `tts_1.5b` 모두 `"enabled": true`로 설정합니다. 서버 실행 시 두 모델 모두 로드됩니다.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvinfer.dll not found` | Add `%TENSORRT_ROOT%\lib` to PATH |
| `cudnn64_8.dll not found` | Add `%CUDNN_ROOT%\bin` to PATH |
| `cudart64_12.dll not found` | Add CUDA bin directory to PATH |
| `trtexec not found` | Set `TENSORRT_ROOT` environment variable |
| `VIBEVOICE_LIB not set` | Set path to VibeVoice Python library before running export_onnx.py |
| Engine build fails with OOM | Use `--no-fp16` or reduce max shapes; close other GPU applications |
| Server starts but TTS returns error | Check `config.json` paths match actual file locations |
| mp3/opus output fails | Place `ffmpeg.exe` in `tools/` directory |

## License

Proprietary. All rights reserved.

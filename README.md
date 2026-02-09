# VibeVoice Windows API Server

Windows Native C++ standalone server providing **OpenAI-compatible TTS/STT REST API** powered by NVIDIA TensorRT.

## System Requirements

| Component | Minimum |
|-----------|---------|
| OS | Windows 10 64-bit or later |
| GPU | NVIDIA GPU with 6 GB+ VRAM (TTS only) or 22 GB+ (all models) |
| CUDA Toolkit | 12.x |
| TensorRT | 8.6+ |
| cuDNN | 8.x (included with CUDA or separate install) |

## Quick Start

### 1. Prepare Models

Run the Python conversion scripts once (requires Python 3.10+ with PyTorch):

```batch
pip install -r scripts\requirements.txt

:: Download tokenizer files
python scripts\prepare_tokenizer.py

:: Export ONNX models
python scripts\export_onnx.py --model-dir <path-to-models> --output-dir onnx

:: Build TensorRT engines
python scripts\build_engines.py --onnx-dir onnx --output-dir engines

:: Convert voice presets (.pt -> .bin)
python scripts\convert_voices.py --voices-dir <path-to-voices>
```

### 2. Configure

Edit `config.json` to match your setup. Key settings:

```json
{
    "server": { "host": "0.0.0.0", "port": 8080 },
    "models": {
        "tts_0.5b": { "enabled": true,  "engine_dir": "engines/tts_0.5b", ... },
        "tts_1.5b": { "enabled": false, ... },
        "asr":      { "enabled": false, ... }
    }
}
```

Set `"enabled": true` only for models whose TensorRT engines are built and available.

### 3. Run

```batch
vibevoice_server.exe --config config.json
```

The server starts on the configured host/port (default `http://0.0.0.0:8080`).

## API Endpoints

### TTS — Text to Speech

```
POST /v1/audio/speech
Content-Type: application/json
```

Request body:
```json
{
    "model": "tts_0.5b",
    "input": "Hello, world!",
    "voice": "alloy",
    "response_format": "wav",
    "speed": 1.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"tts_0.5b"` | `tts_0.5b` or `tts_1.5b` |
| `input` | string | required | Text to synthesize |
| `voice` | string | `"alloy"` | Voice name (see voices endpoint) |
| `response_format` | string | `"wav"` | `wav`, `pcm`, `mp3`, `opus`, `aac`, `flac` |
| `speed` | float | `1.0` | Speech speed multiplier |

Response: binary audio in the requested format.

**OpenAI voice aliases**: alloy, echo, fable, onyx, nova, shimmer are mapped to native voices.

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

## CLI Options

```
vibevoice_server.exe [OPTIONS]

Options:
  --config <path>    Path to config.json (default: config.json)
  --test             Run unit tests and exit
  --test-e2e         Run end-to-end tests and exit
```

## Building from Source

### Prerequisites

- Visual Studio 2022 Build Tools (MSVC v14.x)
- CMake 3.20+
- NVIDIA CUDA Toolkit 12.x
- NVIDIA TensorRT 8.6+
- cuDNN 8.x

### Build

```batch
:: Open VS Developer Command Prompt or run vcvarsall.bat
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Set environment
set TENSORRT_ROOT=C:\path\to\TensorRT
set CUDNN_ROOT=C:\path\to\cudnn

:: Configure and build
cmake --preset release
cmake --build --preset release
```

The executable is produced at `build\release\vibevoice_server.exe`.

### Package for Distribution

```batch
package.bat
```

Creates a `dist\VibeVoiceServer\` directory with the executable, config, and directory scaffolding.

## Directory Structure

```
VibeVoiceServer\
├── vibevoice_server.exe     Server executable
├── config.json              Server configuration
├── engines\                 TensorRT engines (user-built)
│   ├── tts_0.5b\            0.5B streaming TTS engines
│   ├── tts_1.5b\            1.5B full-quality TTS engines
│   └── asr\                 ASR engines
├── onnx\                    ONNX models (intermediate, optional)
│   ├── tts_0.5b\
│   ├── tts_1.5b\
│   └── asr\
├── tokenizer\               BPE tokenizer files
├── voices\                  Voice presets
│   ├── streaming_model\     0.5B voices (.bin)
│   └── full_model\          1.5B voices (.wav)
├── tools\
│   └── ffmpeg.exe           Required for mp3/opus/aac/flac output
└── scripts\                 Model conversion scripts (Python)
```

## Supported Models

| Model | Parameters | Use Case | VRAM |
|-------|-----------|----------|------|
| VibeVoice-Realtime-0.5B | 0.5B | Low-latency streaming TTS | ~2 GB |
| VibeVoice-1.5B | 1.5B | High-quality TTS | ~6 GB |
| VibeVoice-ASR | 7B | Speech recognition / translation | ~16 GB |

## License

Proprietary. All rights reserved.

@echo off
setlocal

set DIST=dist\VibeVoiceServer

echo === VibeVoice Server Packaging ===

:: Clean previous build
if exist "%DIST%" (
    echo Cleaning previous package...
    rmdir /s /q "%DIST%"
)

:: Create directory structure
echo Creating directory structure...
mkdir "%DIST%"
mkdir "%DIST%\engines\tts_0.5b"
mkdir "%DIST%\engines\tts_1.5b"
mkdir "%DIST%\engines\asr"
mkdir "%DIST%\onnx\tts_0.5b"
mkdir "%DIST%\onnx\tts_1.5b"
mkdir "%DIST%\onnx\asr"
mkdir "%DIST%\voices\streaming_model"
mkdir "%DIST%\voices\full_model"
mkdir "%DIST%\tools"
mkdir "%DIST%\scripts"

:: Copy executable
echo Copying executable...
if exist "build\release\vibevoice_server.exe" (
    copy /y "build\release\vibevoice_server.exe" "%DIST%\" >nul
) else (
    echo WARNING: build\release\vibevoice_server.exe not found.
    echo          Run "cmake --build --preset release" first.
)

:: Copy config
echo Copying config.json...
copy /y "config.json" "%DIST%\" >nul

:: Copy tokenizer
echo Copying tokenizer...
if exist "tokenizer" (
    xcopy /e /i /q /y "tokenizer" "%DIST%\tokenizer" >nul
) else (
    mkdir "%DIST%\tokenizer"
    echo WARNING: tokenizer/ directory not found. Run scripts\prepare_tokenizer.py first.
)

:: Copy conversion scripts
echo Copying scripts...
copy /y "scripts\export_onnx.py" "%DIST%\scripts\" >nul
copy /y "scripts\build_engines.py" "%DIST%\scripts\" >nul
copy /y "scripts\convert_voices.py" "%DIST%\scripts\" >nul
copy /y "scripts\prepare_tokenizer.py" "%DIST%\scripts\" >nul
if exist "scripts\requirements.txt" (
    copy /y "scripts\requirements.txt" "%DIST%\scripts\" >nul
)

:: Summary
echo.
echo === Package created: %DIST% ===
echo.
echo Directory structure:
dir /s /b /ad "%DIST%" 2>nul | sort
echo.
echo Files:
dir /s /b "%DIST%" 2>nul | findstr /v /r "\\$" | sort
echo.
echo Next steps:
echo   1. Place ffmpeg.exe in %DIST%\tools\
echo   2. Run model conversion scripts (see scripts\)
echo   3. Edit config.json to match your setup
echo   4. Run vibevoice_server.exe --config config.json

endlocal

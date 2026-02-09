#include "audio/audio_convert.h"

#include <windows.h>
#include <cstdio>
#include <fstream>
#include <string>

namespace AudioConvert {

const char* formatExtension(OutputFormat format) {
    switch (format) {
        case OutputFormat::MP3:  return ".mp3";
        case OutputFormat::OPUS: return ".opus";
        case OutputFormat::AAC:  return ".aac";
        case OutputFormat::FLAC: return ".flac";
    }
    return ".bin";
}

const char* formatCodec(OutputFormat format) {
    switch (format) {
        case OutputFormat::MP3:  return "libmp3lame";
        case OutputFormat::OPUS: return "libopus";
        case OutputFormat::AAC:  return "aac";
        case OutputFormat::FLAC: return "flac";
    }
    return "";
}

// Run ffmpeg as a subprocess with hidden console, wait up to timeoutMs.
static bool runFfmpeg(const std::wstring& cmdLine, DWORD timeoutMs = 30000) {
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi = {};

    // CreateProcessW needs a mutable command line buffer
    std::wstring cmd = cmdLine;

    BOOL ok = CreateProcessW(
        nullptr,
        &cmd[0],
        nullptr, nullptr,
        FALSE,
        CREATE_NO_WINDOW,
        nullptr, nullptr,
        &si, &pi
    );

    if (!ok) {
        fprintf(stderr, "Failed to launch ffmpeg (error %lu)\n", GetLastError());
        return false;
    }

    DWORD waitResult = WaitForSingleObject(pi.hProcess, timeoutMs);
    if (waitResult == WAIT_TIMEOUT) {
        fprintf(stderr, "ffmpeg timed out after %lu ms\n", timeoutMs);
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        return false;
    }

    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    if (exitCode != 0) {
        fprintf(stderr, "ffmpeg exited with code %lu\n", exitCode);
        return false;
    }
    return true;
}

// Get a temp file path with given extension.
static std::wstring getTempFilePath(const wchar_t* ext) {
    wchar_t tmpDir[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmpDir);

    wchar_t tmpFile[MAX_PATH] = {};
    GetTempFileNameW(tmpDir, L"vbv", 0, tmpFile);

    // Replace .tmp extension with desired extension
    std::wstring path(tmpFile);
    size_t dot = path.rfind(L'.');
    if (dot != std::wstring::npos) {
        path = path.substr(0, dot);
    }
    path += ext;
    return path;
}

static std::wstring toWide(const char* s) {
    if (!s || !s[0]) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(len - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], len);
    return w;
}

static bool readFileBytes(const std::wstring& path, std::vector<uint8_t>& out) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                           nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER size;
    GetFileSizeEx(h, &size);
    out.resize((size_t)size.QuadPart);

    DWORD read = 0;
    BOOL ok = ReadFile(h, out.data(), (DWORD)out.size(), &read, nullptr);
    CloseHandle(h);
    return ok && read == (DWORD)out.size();
}

static bool writeFileBytes(const std::wstring& path, const std::vector<uint8_t>& data) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0,
                           nullptr, CREATE_ALWAYS, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;

    DWORD written = 0;
    BOOL ok = WriteFile(h, data.data(), (DWORD)data.size(), &written, nullptr);
    CloseHandle(h);
    return ok && written == (DWORD)data.size();
}

bool convert(const std::vector<uint8_t>& wavData, OutputFormat format,
             std::vector<uint8_t>& output) {
    // Write WAV to temp file
    std::wstring tmpIn = getTempFilePath(L".wav");
    std::wstring tmpOut = getTempFilePath(toWide(formatExtension(format)).c_str());

    if (!writeFileBytes(tmpIn, wavData)) {
        fprintf(stderr, "Failed to write temp WAV file\n");
        DeleteFileW(tmpIn.c_str());
        return false;
    }

    // Build ffmpeg command
    std::wstring cmd = L"ffmpeg.exe -y -i \"" + tmpIn + L"\" -c:a "
                       + toWide(formatCodec(format)) + L" \""
                       + tmpOut + L"\"";

    bool ok = runFfmpeg(cmd);

    if (ok) {
        ok = readFileBytes(tmpOut, output);
        if (!ok) {
            fprintf(stderr, "Failed to read ffmpeg output\n");
        }
    }

    // Cleanup temp files
    DeleteFileW(tmpIn.c_str());
    DeleteFileW(tmpOut.c_str());

    return ok;
}

bool convertFile(const std::string& inputPath, const std::string& outputPath,
                 OutputFormat format) {
    std::wstring wIn = toWide(inputPath.c_str());
    std::wstring wOut = toWide(outputPath.c_str());

    std::wstring cmd = L"ffmpeg.exe -y -i \"" + wIn + L"\" -c:a "
                       + toWide(formatCodec(format)) + L" \""
                       + wOut + L"\"";

    return runFfmpeg(cmd);
}

} // namespace AudioConvert

$env:PATH = "C:\Users\onion\Desktop\Workspace\TensorRT\lib;C:\Users\onion\Desktop\Workspace\TensorRT\bin;C:\Users\onion\Desktop\Workspace\cudnn\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + $env:PATH
Set-Location "C:\Users\onion\Desktop\Workspace\VibeVoiceWindowsApiServer"

# Kill any existing server
Stop-Process -Name vibevoice_server -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Remove old output
Remove-Item -Force "server_stdout.txt" -ErrorAction SilentlyContinue
Remove-Item -Force "server_stderr.txt" -ErrorAction SilentlyContinue

# Start server in background
Write-Host "=== Starting VibeVoice Server ==="
$proc = Start-Process -FilePath ".\build\release\vibevoice_server.exe" -RedirectStandardOutput "server_stdout.txt" -RedirectStandardError "server_stderr.txt" -PassThru

Write-Host "Server PID: $($proc.Id)"
Write-Host "Waiting 20 seconds for engine loading..."
Start-Sleep -Seconds 20

# Check if still alive
if (-not $proc.HasExited) {
    Write-Host "Server is running!"
    Write-Host ""
    Write-Host "=== Server stdout ==="
    Get-Content "server_stdout.txt"
    Write-Host ""
    Write-Host "=== Server stderr (last 20 lines) ==="
    Get-Content "server_stderr.txt" -ErrorAction SilentlyContinue | Select-Object -Last 20
} else {
    Write-Host "Server exited with code $($proc.ExitCode)"
    Write-Host "=== stdout ==="
    Get-Content "server_stdout.txt" -ErrorAction SilentlyContinue
    Write-Host "=== stderr ==="
    Get-Content "server_stderr.txt" -ErrorAction SilentlyContinue
}

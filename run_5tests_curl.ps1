Set-Location "C:\Users\onion\Desktop\Workspace\VibeVoiceWindowsApiServer"

$tests = @(
    @{ name = "English";  json = "test_en.json";    out = "test_en.wav" },
    @{ name = "Japanese"; json = "test_ja.json";    out = "test_ja.wav" },
    @{ name = "Chinese";  json = "test_zh.json";    out = "test_zh.wav" },
    @{ name = "Korean";   json = "test_ko.json";    out = "test_ko.wav" },
    @{ name = "Mixed";    json = "test_mixed.json"; out = "test_mixed.wav" }
)

foreach ($t in $tests) {
    Write-Host "`n=== Test: $($t.name) ==="
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # Use curl.exe to avoid PowerShell encoding issues
    & curl.exe -s -X POST http://localhost:8899/v1/audio/speech `
        -H "Content-Type: application/json; charset=utf-8" `
        -d "@$($t.json)" `
        -o $t.out `
        -w "HTTP_CODE:%{http_code} SIZE:%{size_download}"

    $sw.Stop()
    $dur = [math]::Round($sw.Elapsed.TotalSeconds, 2)

    if (Test-Path $t.out) {
        $size = (Get-Item $t.out).Length
        $audioDur = if ($size -gt 44) { [math]::Round(($size - 44) / (24000 * 2), 2) } else { 0 }
        Write-Host "  File: $size bytes | ${audioDur}s audio | ${dur}s elapsed"
    } else {
        Write-Host "  No output file | ${dur}s elapsed"
    }
}

Write-Host "`n=== Summary ==="
foreach ($t in $tests) {
    if (Test-Path $t.out) {
        $size = (Get-Item $t.out).Length
        $audioDur = if ($size -gt 44) { [math]::Round(($size - 44) / (24000 * 2), 2) } else { 0 }
        Write-Host "  $($t.name): $size bytes (${audioDur}s audio)"
    } else {
        Write-Host "  $($t.name): MISSING"
    }
}

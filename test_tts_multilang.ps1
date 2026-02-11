#
# Multi-Language TTS Test Script
#
# [Voices]
#   공통: carter (default), wayne
#   1.5B Full 전용: alice, frank, maya, mary, samuel, anchen, bowen, xinran (중국어 권장)
#
# [OpenAI 호환 매핑 (4-stage fallback)]
#   alloy->carter, echo->wayne, fable->carter
#   onyx->wayne, nova->carter, shimmer->wayne
#   부분 매치: maya -> en-maya_woman, carter -> en-carter_man
#

param(
    [string]$ApiUrl   = "http://localhost:8899/v1/audio/speech",
    [string]$Model    = "vibevoice-1.5b",
    [string]$Voice    = "maya",
    [string]$OutDir   = "./tts_output"
)

# --- Test definitions ---

$tests = [ordered]@{
    en    = @{ name = "English";  text = "Hello, world! This is a text-to-speech test in English." }
    ko    = @{ name = "Korean";   text = "안녕하세요, 세계! 한국어 음성 합성 테스트입니다." }
    ja    = @{ name = "Japanese"; text = "こんにちは、せかい！にほんごのおんせいごうせいてすとです。" }
    zh    = @{ name = "Chinese";  text = "你好，世界！这是中文语音合成测试。" }
    mixed = @{ name = "Mixed";    text = "Hello! 안녕하세요! こんにちは！你好！This is a multilingual mixed test. 다국어 혼합 테스트입니다. たげんごこんごうてすとです。这是多语言混合测试。" }
}

# --- Prepare output directory ---

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
Get-ChildItem -Path $OutDir -Filter "*.wav" -ErrorAction SilentlyContinue | Remove-Item -Force

# --- Write JSON files (UTF-8 no BOM for curl.exe) ---

$utf8NoBom = New-Object System.Text.UTF8Encoding $false

foreach ($lang in $tests.Keys) {
    $jsonPath = Join-Path $OutDir "test_${lang}.json"
    $body = @{
        model = $Model
        input = $tests[$lang].text
        voice = $Voice
    } | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText((Resolve-Path $OutDir).Path + "\test_${lang}.json", $body, $utf8NoBom)
}

# --- Run tests ---

Write-Host "=== Multi-Language TTS Test ==="
Write-Host "API:   $ApiUrl"
Write-Host "Model: $Model | Voice: $Voice"
Write-Host ""

$results = @()

foreach ($lang in $tests.Keys) {
    $t        = $tests[$lang]
    $jsonFile = Join-Path $OutDir "test_${lang}.json"
    $wavFile  = Join-Path $OutDir "test_${lang}.wav"

    Write-Host "[$($t.name)] $($t.text)"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $curlOut = & curl.exe -s -X POST $ApiUrl `
        -H "Content-Type: application/json; charset=utf-8" `
        -d "@$jsonFile" `
        -o $wavFile `
        -w "%{http_code}|%{size_download}|%{time_total}" 2>&1

    $sw.Stop()
    $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 2)

    # Parse curl -w output
    $parts    = "$curlOut".Split("|")
    $httpCode = if ($parts.Length -ge 1) { $parts[0] } else { "?" }
    $dlSize   = if ($parts.Length -ge 2) { $parts[1] } else { "0" }
    $curlTime = if ($parts.Length -ge 3) { $parts[2] } else { "?" }

    # Check result
    $ok       = $false
    $fileSize = 0
    $audioDur = 0.0
    if ((Test-Path $wavFile)) {
        $fileSize = (Get-Item $wavFile).Length
        if ($fileSize -gt 44) {
            $audioDur = [math]::Round(($fileSize - 44) / (24000 * 2), 2)
            $ok = $true
        }
    }

    if ($ok) {
        Write-Host "  -> HTTP $httpCode | ${fileSize} bytes | ${audioDur}s audio | ${elapsed}s elapsed"
        Write-Host "  -> Saved: $wavFile"
    } else {
        Write-Host "  -> HTTP $httpCode | FAILED (${fileSize} bytes) | ${elapsed}s elapsed"
    }
    Write-Host ""

    $results += [PSCustomObject]@{
        Lang     = $lang
        Name     = $t.name
        Status   = if ($ok) { "OK" } else { "FAIL" }
        HTTP     = $httpCode
        Bytes    = $fileSize
        AudioSec = $audioDur
        Elapsed  = $elapsed
    }
}

# --- Summary ---

Write-Host "=== Summary ==="
Write-Host ("{0,-8} {1,-10} {2,-6} {3,-6} {4,10} {5,8} {6,8}" -f "Lang", "Name", "Status", "HTTP", "Bytes", "Audio", "Time")
Write-Host ("{0,-8} {1,-10} {2,-6} {3,-6} {4,10} {5,8} {6,8}" -f "----", "----", "------", "----", "-----", "-----", "----")
foreach ($r in $results) {
    Write-Host ("{0,-8} {1,-10} {2,-6} {3,-6} {4,10} {5,7}s {6,7}s" -f $r.Lang, $r.Name, $r.Status, $r.HTTP, $r.Bytes, $r.AudioSec, $r.Elapsed)
}

$passed = ($results | Where-Object { $_.Status -eq "OK" }).Count
$total  = $results.Count
Write-Host ""
Write-Host "Result: $passed / $total passed"

# --- List output files ---

Write-Host ""
Write-Host "=== Output Files ==="
Get-ChildItem -Path $OutDir -Filter "*.wav" | ForEach-Object {
    Write-Host ("  {0,-24} {1,10:N0} bytes" -f $_.Name, $_.Length)
}

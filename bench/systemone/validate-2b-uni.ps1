$ErrorActionPreference = 'Stop'
$exe = '.\bench\systemone\eval-9b.exe'
$model = 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf'
$artifact = 'systemone-train/artifacts/qwen38-2b-q4-readout-v2-uni.json'
$data = 'systemone-train/data/generated/splits-gp2/test.jsonl'
foreach ($device in @('gpu', 'cpu')) {
    $log = "bench/systemone/validation-2b-uni-$device.log"
    Write-Output "=== 2B union validation: $device (693 rows) ==="
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # cmd redirects native output directly, avoiding PowerShell UTF-16 logs.
    $command = "`"$exe`" -model `"$model`" -artifact $artifact -data $data -device $device -ctx 512 > `"$log`" 2>&1"
    & cmd.exe /d /s /c $command
    $code = $LASTEXITCODE
    $sw.Stop()
    Write-Output "VALIDATION_DEVICE=$device EXIT=$code SECONDS=$([math]::Round($sw.Elapsed.TotalSeconds, 1)) LOG=$log"
    if ($code -ne 0) { throw "2B $device validation failed; inspect $log" }
    Get-Content $log | Select-String -Pattern 'OVERALL|ZERO-SHOT|SEEN-DOMAIN'
}

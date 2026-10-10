$ErrorActionPreference = 'Stop'
$exe = '.\bench\systemone\eval-9b.exe'
$model = 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf'
$artifact = 'systemone-train/artifacts/qwen35-9b-q3-readout-v2-uni.json'
$data = 'systemone-train/data/generated/splits-gp2/test.jsonl'
$env:DLGO_GEMM_Q3K = '1'
$env:DLGO_SSM_BATCH = '1'
$env:DLGO_SSM_REC_T32 = '1'
$log = 'bench/systemone/validation-9b-uni-gpu-t32.log'
Write-Output '=== 9B union validation: GPU + T32 (693 rows) ==='
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$command = "`"$exe`" -model `"$model`" -artifact $artifact -data $data -device gpu -ctx 512 > `"$log`" 2>&1"
& cmd.exe /d /s /c $command
$code = $LASTEXITCODE
$sw.Stop()
Write-Output "VALIDATION_EXIT=$code SECONDS=$([math]::Round($sw.Elapsed.TotalSeconds, 1)) LOG=$log"
if ($code -ne 0) { throw '9B GPU+T32 validation failed; inspect log' }
Get-Content $log | Select-String -Pattern 'OVERALL|ZERO-SHOT|SEEN-DOMAIN|NLL|BRIER|ECE'

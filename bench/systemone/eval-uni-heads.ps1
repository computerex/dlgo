$ErrorActionPreference = 'Stop'
# Evaluate the union heads on BOTH engines (cpu and gpu) against the 693-row test set.
# Uses the GPU-capable evaluator binary (bench/systemone/eval-9b.exe) which accepts -device.
$test = 'systemone-train/data/generated/splits-gp2/test.jsonl'

Write-Output '=== 2B union head: CPU engine ==='
& .\bench\systemone\eval-9b.exe -model 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf' -artifact systemone-train/artifacts/qwen38-2b-q4-readout-v2-uni.json -data $test -device cpu -ctx 512 2>&1 | Select-String -Pattern 'OVERALL|ZERO-SHOT|SEEN-DOMAIN'

Write-Output '=== 2B union head: GPU engine ==='
& .\bench\systemone\eval-9b.exe -model 'C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf' -artifact systemone-train/artifacts/qwen38-2b-q4-readout-v2-uni.json -data $test -device gpu -ctx 512 2>&1 | Select-String -Pattern 'OVERALL|ZERO-SHOT|SEEN-DOMAIN'

Write-Output '=== 9B union head: CPU engine ==='
& .\bench\systemone\eval-9b.exe -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' -artifact systemone-train/artifacts/qwen35-9b-q3-readout-v2-uni.json -data $test -device cpu -ctx 512 2>&1 | Select-String -Pattern 'OVERALL|ZERO-SHOT|SEEN-DOMAIN'

Write-Output '=== 9B union head: GPU engine ==='
& .\bench\systemone\eval-9b.exe -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' -artifact systemone-train/artifacts/qwen35-9b-q3-readout-v2-uni.json -data $test -device gpu -ctx 512 2>&1 | Select-String -Pattern 'OVERALL|ZERO-SHOT|SEEN-DOMAIN'

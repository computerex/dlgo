Write-Output "=== nvidia-smi ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
Write-Output "=== 2b-short-latency summary ==="
$j = Get-Content -LiteralPath bench/systemone/2b-short-latency.json -Raw | ConvertFrom-Json
Write-Output ("cold_ms p50={0:N1} p90={1:N1} | warm_ms p50={2:N1} p90={3:N1} | speedup={4:N2}x | identical={5}" -f `
  $j.summary.cold_ms.p50, $j.summary.cold_ms.p90, $j.summary.warm_ms.p50, $j.summary.warm_ms.p90, `
  $j.summary.speedup, $j.summary.identical)
Write-Output ("cold_ms keys: " + ($j.summary.cold_ms.PSObject.Properties.Name -join ','))
Write-Output ("summary keys: " + ($j.summary.PSObject.Properties.Name -join ','))

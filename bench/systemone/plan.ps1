nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
Write-Output "=== split row counts ==="
$dir = "C:\projects\dlgo\systemone-train\data\generated\splits-gp2"
foreach ($f in @("train-aug.jsonl","train.jsonl","dev.jsonl","calibration.jsonl","test.jsonl")) {
  $p = Join-Path $dir $f
  if (Test-Path $p) {
    $n = (Get-Content $p | Measure-Object -Line).Lines
    Write-Output ("{0}: {1} lines" -f $f, $n)
  }
}
Write-Output "=== manifest ==="
Get-Content (Join-Path $dir "manifest.json") -Raw

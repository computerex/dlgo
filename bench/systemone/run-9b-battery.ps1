$ErrorActionPreference = 'Stop'
$readout = 'systemone-train/artifacts/qwen35-9b-q3-readout-v1-gpu.json'
Write-Output '=== Yu-Gi-Oh decision battery (9B) ==='
python bench/systemone/evaluate_2b_http.py --url http://127.0.0.1:8097 --model qwen35-9b --readout $readout
if ($LASTEXITCODE -ne 0) { throw '9B battery failed' }
Write-Output '=== HTTP latency (3q short) ==='
python bench/systemone/http_latency.py --url http://127.0.0.1:8097/v1/systemone --request bench/systemone/short-request.json --model qwen35-9b --readout $readout --runs 10 --nonce-seed 9b --output bench/systemone/9b-short-latency.json
Write-Output '=== HTTP latency (3q 10.7K doc) ==='
python bench/systemone/http_latency.py --url http://127.0.0.1:8097/v1/systemone --request bench/systemone/long-request.json --model qwen35-9b --readout $readout --runs 5 --nonce-seed 9b --output bench/systemone/9b-long-latency.json

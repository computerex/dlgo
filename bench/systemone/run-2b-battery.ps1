$ErrorActionPreference = 'Stop'
$readout = 'systemone-train/artifacts/qwen38-2b-q4-readout-v1-gpu.json'
Write-Output '=== Yu-Gi-Oh decision battery (2B) ==='
python bench/systemone/evaluate_2b_http.py --url http://127.0.0.1:8097 --readout $readout
Write-Output '=== HTTP latency (3q short) ==='
python bench/systemone/http_latency.py --url http://127.0.0.1:8097/v1/systemone --request bench/systemone/short-request.json --model qwen38-2b --readout $readout --runs 10 --nonce-seed 2b --output bench/systemone/2b-short-latency.json
Write-Output '=== HTTP latency (3q 10.7K doc) ==='
python bench/systemone/http_latency.py --url http://127.0.0.1:8097/v1/systemone --request bench/systemone/long-request.json --model qwen38-2b --readout $readout --runs 5 --nonce-seed 2b --output bench/systemone/2b-long-latency.json

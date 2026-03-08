# Benchmark Report — 2026-03-07

Commit: `52274d2`

Environment:
- Machine: Intel MacBook Pro with `AMD Radeon Pro 5500M`
- `dlgo GPU`: Vulkan
- `Ollama`: CPU-only on this machine

Settings:
- Prompt: `Write a short poem about the ocean.`
- `max_tokens=64`
- `temperature=0`
- `seed=42`
- 1 warm-up run before the measured run

## Results

| Model | dlgo CPU | dlgo GPU | Ollama CPU | dlgo CPU prefill | dlgo GPU prefill | Ollama prefill |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct-Q4_K_M | 20.40 | 104.20 | 27.40 | 361 ms | 233 ms | 107 ms |
| Phi-4-mini-instruct-Q3_K_M | 6.10 | 22.85 | 10.11 | 1781 ms | 913 ms | 332 ms |
| Qwen3-0.6B-Q8_0 | 22.76 | 105.56 | 34.30 | 179 ms | 248 ms | 63 ms |
| gemma-2-2b-it-Q4_K_M | 9.40 | 43.86 | 11.45 | 733 ms | 488 ms | 299 ms |
| gemma-2b.Q4_K_M | 10.35 | 54.86 | 13.75 | 426 ms | 284 ms | 189 ms |
| gemma-3-1b-it-Q4_K_M | 17.09 | 82.46 | 25.11 | 360 ms | 197 ms | 178 ms |
| gemma-3-270m-it-Q8_0 | 53.87 | 134.72 | 73.95 | 72 ms | 38 ms | 25 ms |
| qwen2.5-0.5b-instruct-q4_k_m | 33.67 | 111.69 | 49.82 | 169 ms | 144 ms | 72 ms |
| smollm2-1.7b-instruct-q4_k_m | 14.64 | 76.04 | 19.69 | 541 ms | 429 ms | 266 ms |
| smollm2-360m-instruct-q8_0 | 36.04 | 121.17 | 53.95 | 137 ms | 84 ms | 75 ms |
| tinyllama-1.1b-chat-v1.0.Q4_0 | 19.69 | 120.84 | 32.63 | 517 ms | 232 ms | 288 ms |
| **AVG** | **22.18** | **88.93** | **32.01** | **480 ms** | **299 ms** | **172 ms** |

## Notes

- `dlgo GPU` output was coherent on the previously broken cases:
  - `Phi-4-mini-instruct-Q3_K_M`
  - `qwen2.5-0.5b-instruct-q4_k_m`
- `Phi` remains the largest GPU prefill outlier even after the fixes.
- `qwen2.5` GPU prefill returned to a normal range after fixing missing `Bq/Bk/Bv` handling in batched prefill.

## Changes Behind This Run

- Fused GPU forward now only runs when all required quantized tensors have native GPU support.
- Batched GPU prefill now applies `Bq/Bk/Bv`.
- Added Vulkan `Q5_K` matvec support, which unblocked the fused path for `Phi`.
- Re-enabled batched GPU prefill for `Phi` after `Q5_K` support landed.

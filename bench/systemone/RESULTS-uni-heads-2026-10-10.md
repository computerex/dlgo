# Unified heads ladder — 0.8B / 2B / 9B (device-any, union-trained)

All three backbone sizes now have a single union-trained readout head (`"device": "any"`)
that runs on both engines. One artifact per size; no per-device heads needed.

## Artifacts (`systemone-train/artifacts/`)

| Size | Artifact | Trained | Backbone | Hidden | Temp |
|---|---|---|---|---|---|
| 0.8B | `qwen35-08b-q4-readout-v10-uni.{json,gguf}` | 2026-09-29 | Qwen3.5-0.8B-Q4_K_M | 1024 | 0.80 |
| 2B | `qwen38-2b-q4-readout-v2-uni.{json,gguf}` | 2026-10-08 | Qwen3.8-2B-Q4_K_M | 2048 | 0.70 |
| 9B | `qwen35-9b-q3-readout-v2-uni.{json,gguf}` | 2026-10-09 | Qwen3.5-9B-Q3_K_M | 4096 | 0.70 |

Training recipe (both 2B/9B): `systemone-train-gpu.exe -device cpu` with
`-feature-cache <cpu cache>` + `-union-cache <gpu cache>` over splits-gp2
(train-aug 6420×2 / dev 708×2 / cal 445×2 / test 693×2 rows), 100 epochs, lr 0.02.
Scripts: `bench/systemone/union-train-{2b,9b}.ps1`.

## Union-test report (both-device test set, 1386 rows)

| Size | Train acc | Dev acc | Cal acc | Test acc | Test NLL | Test Brier |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 69.61% | 76.55% | 77.75% | **78.79%** | 0.557 | 0.311 |
| 2B | 79.98% | 81.99% | 87.75% | **84.49%** | 0.382 | 0.218 |
| 9B | 93.73% | 93.57% | 95.73% | **93.58%** | 0.211 | 0.104 |

## Per-device validation (splits-gp2 test, 693 rows, ctx 512)

### 9B v2-uni (`validation-9b-uni-{gpu,cpu}.log`, 2026-10-09)

| Engine | OVERALL | NLL | Brier | ECE | ZERO-SHOT (517) | SEEN (176) |
|---|---:|---:|---:|---:|---:|---:|
| GPU | **93.65%** | 0.2109 | 0.1045 | 0.0365 | 93.04% | 95.45% |
| CPU | **93.51%** | 0.2112 | 0.1034 | 0.0360 | 92.84% | 95.45% |

Beats the GPU-only v1 head (93.07% GPU, ECE 0.0372). T32-on rerun bit-identical
(93.65%, `validation-9b-uni-gpu-t32.log`).

### 2B v2-uni (`validation-2b-uni-{gpu,cpu}.log`, 2026-10-10)

| Engine | OVERALL | NLL | Brier | ECE | ZERO-SHOT (517) | SEEN (176) |
|---|---:|---:|---:|---:|---:|---:|
| GPU | **84.85%** | 0.3724 | 0.2114 | 0.0304 | 85.11% | 84.09% |
| CPU | **84.13%** | 0.3921 | 0.2237 | 0.0321 | 84.33% | 83.52% |

Beats the GPU-only v1 head on GPU (84.85% vs 84.56%, NLL 0.3724 vs 0.3984, ECE 0.0304 vs 0.0404).
GPU vs CPU spread 0.72 pts. GPU pass 81.6 s, CPU pass 551.9 s (`validate-2b-uni.ps1`).

### 0.8B v10-uni (from earlier evals, 693 rows)

| Engine | OVERALL | ECE |
|---|---:|---:|
| CPU | 78.93% | 0.0389 |
| GPU | 78.64% | 0.0421 |

## Ladder (GPU engine, 693-row test)

0.8B 78.6% → 2B 84.9% → 9B 93.7%
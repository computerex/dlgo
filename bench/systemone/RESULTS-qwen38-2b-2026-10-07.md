# System One on Qwen3.8-2B — results (2026-10-07)

Backbone switched from `Qwen3.5-0.8B-Q4_K_M` to `Qwen3.8-2B-Q4_K_M` (same `qwen35`
family: dim 2048, 24 decoder layers + 1 MTP block, 18 SSM + 6 attention, SSM state
128×128). One GPU-resident server at a time.

## Loader fix (required)
`Qwen3.8-2B` GGUF sets `block_count=25` **including one trailing MTP block**
(`nextn_predict_layers=1`, tensors `blk.24.nextn.*`). The loader previously ran
`blk.24` as an ordinary attention layer → wrong hidden states. Fix: subtract
`<arch>.nextn_predict_layers` from `block_count` in `models/llm/config.go`
(rejects invalid values); regression tests in `models/llm/mtp_test.go`
(parser, synthetic loader, opt-in real-file via `DLGO_TEST_MTP_GGUF`).
Corrected model = **24 layers**.

## Corpus accuracy (splits-gp2 test, 693 rows, GPU engine)
| | 0.8B v10-uni | **Qwen3.8-2B v1-gpu** |
|---|---:|---:|
| Overall | 78.64% | **84.56%** |
| Zero-shot (517) | 78.92% | **84.91%** |
| Seen (176) | 77.84% | **83.52%** |
| ECE | 0.0421 | **0.0404** |
| NLL | — | 0.3984 |
| Brier | — | 0.2249 |

Dev 82.1%/708, calibration 87.9%/445, temperature 0.60.
Zero-shot ≥ seen → meta-task generalization holds at 2B.

Weakest families (2B): medical_urgency 54.6%, reply_urgency 50%, satisfaction 50%,
severity 61.5%, channel/tone 71.4%, code_action 70.5%.

Artifacts:
- `systemone-train/artifacts/qwen38-2b-q4-readout-v1-gpu.{json,gguf}` (device `gpu`, 5×2048, temp 0.60)
- Feature cache `systemone-train/data/generated/gp2-features-cache-2b-gpu-backbone.json`
- Report `systemone-train/artifacts/qwen38-2b-v1-gpu-report.json`

## Yu-Gi-Oh battery (off-corpus, factual)
2B: **6/10** (0.8B: 5/10). Description-matching all correct; factual recall still the
weak axis (missed Blue-Eyes Level-8 tribute rule, Normal-Spell-on-opponent-turn,
Solemn Judgment, and the 3000-ATK score nuance).

## Latency (HTTP end-to-end, GPU, matched nonces, answers bit-identical)
| Workload | 0.8B (batched) | **2B** | ratio |
|---|---:|---:|---:|
| 3q short cold (p50) | 194 ms | 244 ms | 1.26× |
| 3q short warm (p50) | 109 ms | 148 ms | 1.36× |
| 3q 10.7K-doc cold (p50) | 5.35 s | 6.87 s | 1.28× |
| 3q 10.7K-doc warm (p50) | 248 ms | 292 ms | 1.18× |

(2B short bench at ctx 8192; long bench at ctx 16384. `degraded=0`, answers
bit-identical cold vs warm, 5–10 runs each.)

The 2B is only ~1.25× slower than the 0.8B despite 2.5× the parameters — larger
GEMMs are more GPU-efficient, so the per-token cost does not scale with params.

## Build/ops
- Builds must use `-ldflags "-linkmode internal"` (external linker → invalid PE).
- GPU extraction of the 8266-row corpus took ~1.8 h; 2B GPU footprint 1.8 GB at ctx 16384.
- Scripts: `bench/systemone/{build-gpu,train-2b-gpu,serve-2b,run-2b-battery,evaluate_2b_http,meta,readlog}.ps1|py`.

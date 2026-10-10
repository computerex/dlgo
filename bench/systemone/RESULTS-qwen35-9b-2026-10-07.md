# System One on Qwen3.5-9B — results (2026-10-07)

Backbone ladder: 0.8B → Qwen3.8-2B → **Qwen3.5-9B-Q3_K_M** (4.35 GB). Same `qwen35`
family, so this was a drop-in head retrain (no loader changes needed — the 9B GGUF
has `block_count=32` with **no** `nextn_predict_layers`, so no MTP exclusion applies).

## Model facts (verified)
- dim 4096, FFN 12288, 16 heads / 4 KV heads, key_len 256
- `full_attention_interval=4` → 8 attention + **24 SSM** layers
- `ssm.inner_size=4096`, `time_step_rank=32` → SSM head V dim **128** (exactly at the
  batched-path limit — `DLGO_SSM_BATCH` fast prefill works)
- GPU residency: **all 32 layers, 4,515 MB VRAM**

## Corpus accuracy (splits-gp2 test, 693 rows, GPU engine)
| | 0.8B v10-uni | Qwen3.8-2B | **Qwen3.5-9B** |
|---|---:|---:|---:|
| Overall | 78.64% | 84.56% | **93.1%** |
| NLL | — | 0.3984 | **0.2208** |
| Brier | — | 0.2249 | **0.1111** |
| Dev | — | 82.1% | 93.4% |

Training: 1.71 h GPU extraction+train (8,266 rows), temp 0.65, 5×4096 readout.

Artifacts:
- `systemone-train/artifacts/qwen35-9b-q3-readout-v1-gpu.{json,gguf}` (device `gpu`)
- Feature cache `systemone-train/data/generated/gp2-features-cache-9b-gpu.json`
- Report `systemone-train/artifacts/qwen35-9b-v1-gpu-report.json`

## Zero-shot / seen breakdown
| | Overall | Zero-shot (517) | Seen (176) | ECE |
|---|---:|---:|---:|---:|
| 0.8B v10-uni | 78.64% | 78.92% | 77.84% | 0.0421 |
| Qwen3.8-2B | 84.56% | 84.91% | 83.52% | 0.0404 |
| **Qwen3.5-9B** | **93.07%** | **92.65%** | **94.32%** | **0.0372** |

Weakest 9B families: satisfaction 50% (4 rows), medical_urgency 65.5%, severity
69.2%, code_action 75.0%. Everything else ≥85%.

## Yu-Gi-Oh battery (off-corpus, factual)
| | 0.8B | 2B | **9B** |
|---|---:|---:|---:|
| Score | 5/10 | 6/10 | **9/10** |

The 9B now correctly answers Normal-Spell-on-opponent-turn (no @ 0.022), Solemn
Judgment (negate by paying half LP), and the 3000-ATK score (idx 3). Only remaining
miss: Blue-Eyes Level-8 Normal Summon needs 2 Tributes (answered yes @ 0.588 —
nearly a coin flip, i.e. honestly uncertain rather than confidently wrong).

## Latency (HTTP end-to-end, GPU, matched nonces, answers bit-identical)
| Workload | 0.8B | 2B | **9B** |
|---|---:|---:|---:|
| 3q short cold (p50) | 194 ms | 244 ms | **2,558 ms** |
| 3q short warm (p50) | 109 ms | 148 ms | **1,546 ms** |
| 3q 10.7K-doc cold (p50) | 5.35 s | 6.87 s | **87.85 s** |
| 3q 10.7K-doc warm (p50) | 248 ms | 292 ms | **1.84 s** |

9B: 10 short samples and 5 long samples; all cold/warm answer maps identical,
no degraded questions. Long-document p95: 89.23 s cold, 1.85 s warm.
These are historical cross-model comparisons, not identical-nonce matched model
comparisons. Corpus leakage/model-selection caveats documented in
`docs/systemone-parity-plan.md` also apply; the zero-shot group is not an untouched
model-selection holdout.

**Why the 9B is ~10× the 2B rather than ~4.5×** — profiled with `DLGO_SSM_TIME=1`
(batched path **confirmed engaged**): a ~120-token request spent
**FFN 958.7 ms (75%)**, SSM 224.9 ms, attn 86.7 ms, embed 4.5 ms.
The 9B's FFN (dim 4096 → 12288, mixed Q3_K/Q4_K weights) dominates this profile.
This identifies the FFN kernels as an optimization target, but does not isolate
quantization format from model size or prove a specific TFLOPS rate. A same-backbone
Q3-versus-Q4 benchmark is needed before attributing the slowdown to Q3_K alone.
The accuracy/latency trade is real: this backbone buys +8.5 pts corpus accuracy and
+3 Yu-Gi-Oh points for ~10× decision latency vs the 2B.

## Ops
- One server at a time; VRAM verified clean (14.1 GB free) before onboarding.
- Scripts: `bench/systemone/{build-9b,smoke-9b,train-9b-gpu,serve-9b,run-9b-battery,meta9b}.ps1`.

# dlgo System One / Jev-Compatible Decision Model — Work Summary

Last updated: 2026-09-28 (v8-gpu: shared-prefill multi + GPU decision engine complete)

## Goal
Build a production-grade, Jev-like, general-purpose decision model system in the `dlgo` Go project. Fast, cheap, local decision/classification that works like TypeSafe's Jev (state + typed questions → typed answers + probabilities), on any supported small LLM, built from scratch (no reuse of Jeff/AutoJev).

## Hardware / Environment
- RTX 4070 Ti SUPER 16GB, Python 3.12, PyTorch 2.6 CUDA 12.4
- Local Q4 GGUF: `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`
- DeepSeek API key in `systemone-train/.env` (gitignored)

## Architecture Decision
- **Primary engine = direct finite-choice likelihood scoring** (shared prefill + candidate continuation scoring)
- **Native decision-readout** (one forward pass → final hidden state → trained linear classifier head → calibrated softmax) is the target for decision-tuned models
- **Q4-based fine-tuning**: train a new linear readout head on frozen Q4 GGUF backbone features (cannot modify packed Q4 weights)
- Generic continuation scoring retained as fallback for ordinary LLMs
- Jev-like API: `choice`, `noul`, `score` primitives; `/v1/systemone` endpoint

## What Is Implemented (working)
- **System One core** (`models/llm/systemone.go`): `SystemOne()`, `SystemOneHidden()`, `SystemOneNoul()`, `SystemOneScore()`, `SystemOneMulti()` (shared prefill), prompt compiler (`CompileSystemOnePrompt`, `compileSystemOneParts`), continuation scoring, probability normalization, stable value-derived answer codes, question builders (`BuildSystemOneNoulQuestion`, `BuildSystemOneScoreChoices`)
- **Native readout** (`models/llm/systemone_readout.go`): `SystemOneReadout` (with `Device` field), `LoadSystemOneReadout()`, `LoadSystemOneReadoutGGUF()`, `SystemOneWithReadout()`, `ApplySystemOneReadout()` (engine-agnostic scoring of pre-extracted features)
- **ForwardHidden** (`models/llm/forward.go`): skip full-vocab projection for readout path
- **KV/SSM cloning** (`memory/clone.go`): `MultiLayerKVCache.Clone()`, `SSMStateCache.Clone()`, `SSMLayerState.Clone()`
- **Grammar fix** (`models/llm/generate.go`): public Generate/Chat now honor `WithGrammar`; KV cache crash fix for Qwen3.5 hybrid
- **GGUF writer** (`format/gguf/writer.go`): `WriteGGUF()` serializes readout tensor + metadata as standalone GGUF (round-trip tested)
- **Trainer** (`cmd/systemone-train/main.go`): linear readout on frozen Q4 features, train/dev/cal/test splits, dev-based checkpoint selection, temperature fitting on calibration, metrics (accuracy/NLL/Brier), feature cache (device-keyed), `-device gpu` support (`device_gpu.go`/`device_stub.go`), GGUF export with `systemone.device` metadata
- **Evaluator** (`cmd/systemone-eval/main.go`): per-family metrics + **ECE** (Expected Calibration Error), reliability diagram, zero-shot vs seen-domain gate, `-device` flag
- **CLI** (`cmd/dlgo/systemone.go`): `dlgo systemone` command
- **Server** (`server/systemone.go`): `POST /v1/systemone` endpoint with readout caching, per-question graceful degradation, `engine` field, **scheduler-serialized** (mutually exclusive with generation via `Scheduler.WithInferenceLock`), GPU-preferred with CPU fallback
- **GPU decision engine** (`gpu/pipeline.go`, `gpu/forward.go`): `GpuPipeline.SystemOneHidden()` (batched prefill, vocab projection skipped, hidden downloaded from rs.X post-output-norm), `SystemOneWithReadout()`, `SystemOneMulti()` (per-question GPU prefill, wide questions degrade to CPU), device gates (refuses cpu-device readouts), `hiddenBuf` parameter on both batch prefill functions
- **Data pipeline** (`systemone-train/`): `generate_data.py` (DeepSeek synthetic, batch-aware IDs), `split_data.py` (deterministic split by ID hash, dedup)

## Training Results
### v5 (previous best) — 1481-example corpus, 6 families
- Families: support_route, moderation, sentiment, intent, urgency, topic
- Splits: train 1044, dev 232, calibration 104, test 101
- **dev 89.2%/232, test 90.1%/101, NLL 0.281, Brier 0.160, ECE 0.037, temperature 0.35**
- Per-family test accuracy: intent/moderation/sentiment/support_route 93.75%, topic 88.9%, urgency 78.9%
- Reliability: [0.9-1.0) bin conf=0.974 vs acc=0.971 (gap +0.003, n=69) — near-perfect calibration
- Artifact: `systemone-train/artifacts/qwen35-08b-q4-readout-v5.json`

### v6 (permutation-augmented, 6 support families)
- Train: 2088 rows (1044 + 1044 code-permuted variants); dev/cal/test untouched
- dev 86.2%/232, test 85.1%/101, NLL 0.383, Brier 0.216, temp 0.45
- **Key evidence of meta-task learning**: on hand-crafted e2e cases, v6 fixed the billing case (selected billing 0.561 instead of account 0.959) and collapsed overconfidence. Technical case still wrong but distribution much flatter.
- First GGUF export produced: `qwen35-08b-q4-readout-v6.gguf`

### v7-gp (GENERAL-PURPOSE, current best) — domain-agnostic corpus
- **20 domains**: 14 trained + **6 zero-shot-only** (medical_urgency, legal_domain, code_action, financial_type, travel_status, education_level)
- Random 2–5 label subsets per example (from per-domain pools) so domain mappings cannot be memorized; 1425 permutation-augmented training rows; dev 185, cal 112, test 188
- **ACCEPTANCE GATE PASSED — ZERO-SHOT domains (never trained): 86.1% accuracy (137 test rows)** vs seen-domain held-out 68.6% (51 rows, 1–5 per family — high variance)
- Zero-shot family detail: travel_status 100%, education_level 95.2%, legal_domain 95.0%, medical_urgency 79.3%, code_action 77.3%, financial_type 75.0%
- Overall test 81.4%/188, NLL 0.518, Brier 0.283, ECE 0.0585, temp 0.60; readout 5×1024 (supports up to 5 labels)
- Artifacts: `qwen35-08b-q4-readout-v7-gp.json` + `.gguf` (GGUF export exercised end-to-end)
- **Conclusion: the readout learned the domain-agnostic label-matching task, not domain mappings — the System One engine is general-purpose in the Jev sense.**

### v8-gpu (GPU-NATIVE, current best for GPU serving)
- Same GP corpus, features extracted **on the RTX 4070 Ti SUPER via Vulkan** (`-device gpu`)
- **dev 76.8%/185, test 76.6%/188, NLL 0.6239, Brier 0.3459, ECE 0.0559, temp 0.35**
- **ZERO-SHOT domains (never trained): 81.75% (137 rows)** — travel_status 90.5%, education 81.0%, legal_domain 95.0%, code_action 86.4%, financial_type 70.8%, medical_urgency 65.5%; seen-domain 68.6% (51 rows)
- Artifacts: `qwen35-08b-q4-readout-v8-gpu.json` + `.gguf` (device=gpu in JSON + GGUF metadata)
- **Why device-native readouts are required**: GPU and CPU extract structurally different hidden states (GPU matvec kernels quantize activations to Q8_1; probes with DLGO_F32_KV=1 and DLGO_NO_DP4A=1 confirmed the ~0.97 CPU-vs-GPU hidden cosine is structural, not KV/precision-switchable). Serving the CPU-trained v7 on GPU gave 2/6 decision agreement (unusable). Retraining on GPU features restores in-distribution accuracy (78.8% zero-shot measured in-test, 145ms/question).
- **Device tax**: ~3–4pt accuracy vs CPU-native (76.6% vs 81.4% test) — the price of Q8_1 activation quantization; in exchange, ~3–4× latency win (135–235ms vs 490–850ms per question on CPU)
- Run-to-run GPU variance: ~±2pt overall on borderline seen-domain rows (within-process cosine spread ~0.003); zero-shot counts were identical across runs

### Known limitation
- The backbone has an off-distribution prior favoring earlier answer codes (A > B > C). Permutation augmentation largely removed the readout's reliance on code identity; the two remaining hand-crafted support-routing cases now sit near coin-flip (account 0.50–0.57) — the confidence honestly reflects ambiguity instead of hiding it behind 0.96 overconfidence.

### gp-v2 corpus (2026-09-29) — scaled 2.4×, basis for v9
- **3456 rows / 36 domains (24 trained + 12 zero-shot-only)** via 216 DeepSeek requests; new zero-shot domains: academic_field, energy_source, music_genre, nutrition, security_threat, transport_mode
- Splits (`splits-gp2`): train 1605 → **6420 augmented** (3 permutation variants), dev 708, cal 445, test 693 (manifest sha256 37da5661…)
- v9 CPU + GPU readouts training on this corpus (fresh per-device feature caches)

### v9-cpu + v9-gpu (SCALED CORPUS gp-v2, current best for both devices)
- Corpus **gp-v2**: 3456 rows / 36 domains (24 trained + **12 zero-shot-only**; new zero-shot families: academic_field, energy_source, music_genre, nutrition, security_threat, transport_mode); splits-gp2: train 6420 augmented (3 permutation variants), dev 708, cal 445, **test 693 (517 zero-shot + 176 seen)**
- **v9-cpu** (`qwen35-08b-q4-readout-v9-cpu.{json,gguf}`): dev 76.4%/708, **test 76.91%/693**, NLL 0.5733, Brier 0.3184, ECE 0.0429, temp 0.70; zero-shot 78.14% (517), seen 73.30% (176)
- **v9-gpu** (`qwen35-08b-q4-readout-v9-gpu.{json,gguf}`, device=gpu): dev 76.0%/708, **test 78.07%/693**, NLL 0.5785, Brier 0.3181, **ECE 0.0331**, temp 0.70; zero-shot 78.92% (517), seen 75.57% (176)
- **Device tax FLIPPED at scale**: with 4.5× more training data the GPU-native readout now BEATS the CPU-native one on the same corpus (78.07% vs 76.91%) and is better calibrated (ECE 0.0331 vs 0.0429) — GPU is now both faster (3–4×) and more accurate. Earlier v7 (CPU-trained) on GPU remained unusable (2/6 agreement), so device-native training is still mandatory.
- Zero-shot families (GPU eval): academic_field 97.6% (41), legal_domain 97.4% (38), energy_source 93.6% (47), education_level 89.8% (49), travel_status 87.5% (40), security_threat 85.0% (40), music_genre 75.0% (44), transport_mode 75.8% (33), code_action 70.5% (44), financial_type 61.4% (44), medical_urgency 61.8% (55), nutrition 57.1% (42)
- Weakest families both devices: format ~43%, satisfaction 25% (4 rows), financial_type ~59–61% (expense↔income↔transfer confusion), medical_urgency ~62%, nutrition ~57%, risk ~57%, severity ~54–62%
- Zero-shot ≥ seen on BOTH devices at scale — meta-task (label-matching) generalization confirmed on 36 domains

### v10-uni (DEVICE-AGNOSTIC — one artifact for both engines, current recommended default)
- Motivated by "why two artifacts?": trained ONE head on the **union of CPU+GPU features** (both caches existed — zero new extraction via the trainer's `-union-cache` mode). Artifact carries `device:"any"`, accepted by both engines.
- **CPU eval: test 78.93%/693, ECE 0.0389, zero-shot 79.69%, seen 76.70%** — beats v9-cpu (76.91%) by +2.0pts
- **GPU eval: test 78.64%/693, ECE 0.0421, zero-shot 78.92%, seen 77.84%** — beats v9-gpu (78.07%) by +0.6pts
- The union didn't trade accuracy for portability — it *improved* it (2× training rows; device drift acts as augmentation). Artifacts: `qwen35-08b-q4-readout-v10-uni.{json,gguf}`, temp 0.80
- `device:"any"` plumbing: loader validation accepts cpu/gpu/any; CPU engine refuses only "gpu" artifacts; GPU engine accepts "gpu"+"any"; server prefers GPU for "gpu" and "any"; systemone-eval `-device` accepts either device for "any" artifacts (defaults to cpu when empty)
- **Fixed a latent trainer bug found on the way**: `featureCacheKey` iterated the `Codes` map, so Go's randomized map iteration made keys for code-augmented rows differ run-to-run (~30–40% silent cache misses). Keys are now canonical (sorted codes); lookups tolerate legacy rotations via `featureCacheKeys`/`lookupFeature` (regression test added). Union mode also initializes Vulkan just to resolve the device name for gpu-cache key suffixes (`ensureGPUCacheSuffix`).

### Pokemon zero-shot battery (2026-09-29, live servers, v10-uni) — generalization + perf check
- Fixtures: `systemone-train/request-pokemon-{multi,fire,electric,noul}.json` — Pokemon battle states with factual type-chart ground truth; completely off-corpus domain
- **Decisions consistent across engines (same answers CPU & GPU): 4/6 factual correct** — multi: switch→pikachu (Electric vs Water ✓), move_type→electric ✓, noul "Charizard weak to Water?"→yes ✓; singles: fire-swap→arcanine ✗ (picked the resist over Water's super-effective+resist), electric-swap→butterfree ✗ (picked the Electric-weak option over Ground immunity), noul "Fire super-effective vs Grass?"→yes ✓
- Misses chose plausible-but-suboptimal options with high confidence (66%/77%) — long multi-sentence criteria on an off-corpus domain is the hard case for a linear probe; matches the "corpus is the truth, hand batteries are harder" finding
- **Latency (warm, HTTP end-to-end): GPU multi (4 questions, 181-token prompt) 0.51s; GPU singles 0.21–0.58s** vs **CPU multi 3.62s, CPU singles 0.85–1.44s → ~2.5–7× GPU speedup**
- Degraded score question (0–10, 11 buckets): GPU-resident server returns uniform abstained+degraded placeholder (CPU continuation scorer unavailable — buffers freed by FreeForGPU); CPU server runs real continuation scoring → gave 6.0/10 for "keep Fire-type in vs Water" (miscalibrated mid — GPU's honest abstain is the better behavior)
- **Server fixes made**: GPU-resident + over-width questions no longer 500 (uniformDegradedResult); cpu-device readout on GPU-resident model → clear 503; degraded flag now rendered for score/noul answers; single-question `prompt` field accepted as alias of `instructions`
- Ops note: dlgo-server defaults to the model's native 262144 ctx — two instances exhaust Windows commit (errno 1455 OOM); pass `-ctx 8192` for decision workloads. Build GPU server with `-ldflags "-linkmode internal"` (external linker produced invalid PE)

### History
- v5 (1481 examples, 6 support families): dev 89.2%/232, test 90.1%/101, NLL 0.281, ECE 0.037
- v4 (1192 examples): dev 88.1%/185, test 86.7%/83, NLL 0.341, Brier 0.196, ECE 0.082
- v3 (1192 examples, pre-dedup splits): dev 84.8%, test 90.5%/21 (small test set)
- v2 (359 examples): dev 76.5%, test 25%/4 (too small)
- v0/v1 (15 seed examples): overfit

## Server Verified Live (v7-gp)
All primitives exercised against `POST /v1/systemone` with the general-purpose readout:
- **Zero-shot choice** (legal landlord dispute, never-trained domain): housing @ 93.4%
- **Zero-shot choice** (medical, borderline case): self_care @ 48.8% vs urgent_care 41.4% — correct, honestly uncertain
- **noul**: "charged twice" → yes @ 72.3%
- **score** (0–10, 11 buckets > 5 readout slots): graceful degradation to continuation scoring, urgency 10 @ 56.9%, `degraded:true` reported
- **multi** (3 questions, one request): billing @ 98.4%, noul yes @ 77.0%, zero-shot severity minor @ 68.5%
- Server now degrades gracefully when a request exceeds readout option capacity (`degraded` field) instead of erroring
- GPU serving: use the gpu-device readout fixture `systemone-train/request-multi-live-gpu.json` (v8-gpu) — responses carry `engine:"gpu"`; CPU fallback automatic on any GPU error or device mismatch

## Current Work In Progress
1. **Shared-prefill multi-question (DONE)** — `SystemOneMulti` prefills the shared prefix once; per-question suffix runs from cloned KV/SSM branches with token-boundary safety (BPE-unsafe questions fall back to standalone). Exact E2E parity vs standalone (readout + fallback). Benchmarks (3 questions, warm): standalone-readout 1804ms → multi-readout 1252ms (**1.44×**), standalone-fallback 1855ms → multi-fallback 1259ms (**1.47×**); speedup grows with question count. Per-question graceful degradation (`Degraded` flag) flows through the server.
2. **GPU (Vulkan) decision engine (DONE)** — `GpuPipeline.SystemOneHidden/SystemOneWithReadout/SystemOneMulti`: batched prefill with `hiddenBuf` replaces the vocab projection with a direct hidden download; readout applied CPU-side (`ApplySystemOneReadout`). Device-aware artifacts (`Device` field in JSON + GGUF metadata, gates both directions) with v8-gpu trained on GPU features. Server prefers GPU when the readout is gpu-device and falls back to CPU on any GPU error; `engine` field reports which engine served. Latency: **135–235ms/question GPU vs 490–850ms CPU (3–4×)**; 4-question multi 1245ms (1 degraded) vs 1145ms CPU 3-question shared-prefill multi.
3. **Concurrency fix (DONE)** — `/v1/systemone` now runs under `Scheduler.WithInferenceLock`, mutually exclusive with the generation loop (shared KV/run-state/GPU command buffers made concurrent evaluation unsafe).
4. **Cost diagnostics (DONE)** — `systemone_diag_test.go`: engine ~6–7ms/token CPU (no big fixed overhead: 2-token ForwardBatch 26.5ms); KV+SSM clone 5.3ms; shared prefix prefill 326ms vs full 629ms (~100-token prompt).

## Tests Passing
`go test ./models/llm ./memory ./grammar ./server ./cmd/dlgo ./cmd/systemone-train ./cmd/systemone-eval ./format/gguf` all pass.
Vulkan-tag builds (`-tags "cgo vulkan"`) compile for `./gpu ./cmd/dlgo-server ./cmd/dlgo ./cmd/systemone-train ./cmd/systemone-eval`; GPU E2E tests (`DLGO_SYSTEMONE_E2E=1`): hidden parity (cosine ~0.97–0.98 vs CPU), device gates, device metadata (JSON+GGUF), decision battery, **GPU corpus accuracy gate (74.5–76.6% observed, zero-shot 78.8%)**, GPU multi with degradation.
Full `go test ./...` blocked by pre-existing unrelated failures (bench/scripts multiple mains, Vulkan-only symbols in cmd/test_glm, examples/json_grammar_gpu).

## Next Steps
1. Scale the GP corpus (more domains, more examples/domain) to push zero-shot accuracy higher on both devices
2. GPU shared-prefill multi (requires GPU KV/SSM branch support — SSM state buffers are baked into per-layer GPU buffers; GPU multi is already ~2× CPU multi, so deferred)
3. Embed readout in the backbone GGUF file (loader support)
4. Consider F32-activation GPU kernels for decision workloads to shrink the device tax
5. Update `jev-work-checklist.md` as items complete

## Constraints
- Do NOT reuse Jeff/AutoJev code, weights, datasets, templates
- Do NOT modify packed Q4 backbone weights — only train the readout head
- DeepSeek key must stay out of source/logs/git; only in `systemone-train/.env`
- GGUF is inference-only; training uses hidden features from frozen Q4 backbone

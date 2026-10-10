# System One latency and accuracy parity plan

> **Update 2026-10-07:** Priority-1 latency work landed — batched SSM prefill, recurrent-only branch restore, cancellation, and timing headers. Measured results: `bench/systemone/RESULTS-2026-10-07.md` (2–2.9× end-to-end, answers bit-identical; short-state requests now 125–208 ms cold / 32–136 ms warm on RTX 4070 Ti SUPER). Remaining latency levers: chunked parallel delta-rule scan for cold long-doc prefill (~6 s), and batching the per-question suffix passes.

## Review scope

Reviewed the current CPU/GPU decision paths, HTTP serving, readout loader, trainer, evaluator, saved training reports and GDPR replay results. This is an assessment, not a claim that parity has been achieved. No vendor API calls or fresh model E2E benchmarks were performed. Existing working-tree changes were left intact.

Validation: `go test ./models/llm ./server ./cmd/systemone-train ./cmd/systemone-eval` passed. These default tests do not establish GPU numerical parity or model accuracy.

## Current evidence

- Architecture: frozen Q4 backbone, final hidden state, L2 normalization, linear answer-code readout, temperature-calibrated masked softmax. The native path avoids vocabulary projection and text generation.
- `systemone-train/artifacts/v10-uni-report.json`: 1,092/1,386 correct (78.79%), NLL 0.5571, Brier 0.3109, temperature 0.8. Union evaluation contains two feature distributions for the same 693 examples, not 1,386 independent examples. `work-summary.md` records CPU 78.93%, GPU 78.64%; these are historical measurements, not fresh results.
- `.sessions/parallel_results_v2.json`: roughly 17.8–17.9 seconds for 13 questions on a roughly 10.7K-token document. These saved results predate or otherwise do not demonstrate the current prefix-cache path. Do not use them as current cache-hit measurements.
- Current `gpu/pipeline.go` already includes a four-entry exact-prefix snapshot cache. Prefix caching is not a missing implementation; cache-hit performance and correctness coverage are missing evidence. No prefix-cache-specific tests were found in the GPU test files searched.
- `gpu/forward.go` still loops over every token in each SSM layer, with separate copies, dispatches, barriers and periodic submit/wait. Matrix projections are batched, recurrence is not.
- `GpuPipeline.SystemOneMulti` shares prefill, then runs suffix branches sequentially. It is not parallel question execution.
- GPU-resident HTTP serving returns uniform, degraded, abstained placeholders for questions wider than `readout.MaxOptions`. This is not continuation scoring. CPU continuation scoring exists, but CPU buffers have been freed on GPU-resident models. Compatibility documentation and old comments overstate this behavior.
- The linear code decoder supports A–Z only. The HTTP option limit is not the trained/native capacity; widening to 255 options requires a new code representation or different scorer.
- `handleSystemOne` holds the inference lock through evaluation and response writing; request cancellation is not passed to the decision engine.

## Priority 0: establish a defensible parity benchmark

Use the same independently labeled cases with dlgo, a pinned Jev version and a pinned OpenAI Decisions model. Vendor outputs are comparators, not ground truth. Preserve rubric meaning in provider adapters; scores must use the same scale and uncertainty semantics.

Test dimensions:

- State lengths: approximately 128, 512, 2K, 8K and 16K tokens; also retain byte length because tokenizers differ.
- Independent questions: 1, 4, 13 and 32.
- Option counts: 2, 5, 10, 26 and wide-option cases where providers support them.
- Short labels versus paragraph-length rubric descriptions.
- Unique states (cold prefix), exact repeated states (warm prefix), and interleaved states causing cache eviction.
- Concurrency 1 and representative production concurrency; mixed generation/decision traffic separately.
- Negation, exceptions, numeric thresholds, conflicting evidence, long-document retrieval, reordered labels, renamed option IDs, missing evidence and prompt injection in state.

Record client E2E p50/p95/p99, queue wait, tokenization, shared prefill, snapshot/restore, suffix compute, readout, cache hit, native/degraded coverage, hardware, binary revision and artifact/dataset hashes. Compare categorical accuracy/macro-F1, noul NLL/Brier, score MAE, calibration and error rate at matched abstention coverage. Count uniform overflow placeholders as unserved coverage, not native successful decisions. Use paired confidence intervals grouped by independent state/example, not device rows or augmentation variants.

An initial parity gate can be accuracy no more than 2 percentage points below either comparator and E2E p95 within 1.25x on agreed workloads; these are proposed product targets, not vendor guarantees. Evaluate cold and warm paths independently.

## Priority 1: cold-prefill latency

### 1. Profile before tuning

Instrument dispatch/submit counts and timings. Measure current prefix-cache cold/hit/eviction behavior. Do not extrapolate old 18-second results to the current binary.

### 2. Replace token-by-token SSM prefill

Implement a gated-delta-rule chunked algorithm rather than merely increasing flush interval. Batch causal convolution and preprocessing; use chunk-local delta-rule computation plus propagation of the incoming recurrent state. A naive elementwise prefix sum is not equivalent to this matrix-valued recurrence.

A lower-risk intermediate experiment is fusing decay/predict/update/output per head/value tile, preserving required synchronization and reduction semantics. Measure whether dispatch savings justify the kernel before committing to the more complex chunked implementation.

Validation must compare hidden states, recurrent state, logits and decisions against the sequential reference across chunk boundaries, partial chunks, prefix restore and long contexts. Floating-point reordering may not be bit-exact: report tolerance and accuracy drift, not just cosine similarity. Keep a reference fallback.

### 3. Parallelize question suffixes

Add true batched branches with independent SSM/conv state and immutable shared-prefix KV plus branch-local suffix KV. Do not concurrently drive the existing pipeline's shared mutable buffers. Benchmark memory, restore cost and p95 as question count grows.

### 4. Harden serving

Propagate request cancellation through queue acquisition and prefill chunk/question boundaries. Separate response serialization/writing from the inference critical section. Expose cache-hit and queue timing; make snapshot cache VRAM-budgeted instead of relying solely on four entries. Add cached/fresh parity, eviction, teardown and mixed generation/decision tests.

## Confirmed benchmark and training weaknesses

A read-only dataset/cache audit reproduced v10 CPU 547/693 (78.93%) and GPU 545/693 (78.64%). It also found:

- Individual-row splitting allows 106 generation batches to occur in both training and test. Generation chooses label subsets and target emphasis per batch. A state-blind training-majority baseline on exact label subsets gets 107/176 (60.80%) on seen-domain test rows. This demonstrates a shortcut opportunity, not proof the head exploits it.
- Reserved “zero-shot” domains are absent from SGD but present in development (367 rows) and calibration (268 rows). They are not untouched by model selection.
- All 693 test examples use alphabetical answer-code mappings. Current evaluation does not establish arbitrary mapping/presentation invariance.
- `augment_data.py` initializes its permutation set with label values instead of code letters. GP2's 6,420 training rows contain only 5,082 distinct state/code combinations; 2,686 retain the alphabetical mapping. These are training duplicates, not train/test leakage.
- No exact train/test state+instructions+criteria overlaps were found. One near-exact train/dev state differs only by capitalization. Generation-batch dependence remains the larger concern.
- Feature-cache keys omit criteria descriptions and prompt version, and identify the model by path rather than content. Label-description changes can silently reuse incompatible features. Current cache staleness was not established.
- Generated corpus quality evaluation is choice-only and synthetic; it does not establish noul/score, long-context, or production calibration quality.

Before retraining, fix permutation uniqueness, make feature identities include complete compiled prompt/backbone/extraction fingerprints, and establish batch-grouped splits plus domains excluded from training, development and calibration. Preserve old reports for comparison, but do not interpret a lower score on the cleaner benchmark as a regression by itself.

## Priority 2: accuracy experiments

1. Freeze the present split and avoid using test errors for checkpoint/hyperparameter selection. Build a new independent, human-audited evaluation set before claiming parity.
2. Reuse CPU+GPU feature caches for dev-selected optimizer/regularization/seed sweeps and a small nonlinear head ablation. Compare NLL and per-family accuracy, not just aggregate accuracy. The head is cheap; the backbone prefill dominates latency.
3. Distinguish head limitations from backbone limitations: compare the linear head, a small MLP, and answer-code likelihood scoring on the same labeled dev cases. A larger head cannot recover evidence absent from the hidden representation.
4. Expand training coverage for rubric-heavy questions, negation, numeric/financial distinctions and long contexts. Audit synthetic labels first. Generate balanced answer-code assignments, rename option IDs, vary option order and randomize rubric phrasing; test invariance independently.
5. Native capacity: train at least 10-option coverage for the complete Jev score rubric range. Existing five-option feature caches cannot validate newly added output slots. Support for 26+ options requires revisiting the A–Z answer-code representation and training data, not padding the weight matrix.
6. If frozen-feature heads plateau, evaluate a stronger frozen backbone with new caches. Backbone adapter training is a separate scope decision; do not modify packed Q4 weights or reuse Jeff/AutoJev material.
7. Calibrate only on held-out calibration data; report calibration by task, option count and context length. The Jev distribution-shape confidence statistic is not a calibrated correctness probability.

## External references

- TypeSafe introduction: https://typesafe.ai/blog/introducing-system-one-models-and-jev — search results advertise 70–500ms end-to-end, not a guaranteed matched-workload SLO.
- OpenAI Decisions guide: https://developers.openai.com/api/docs/guides/decisions — search results describe typed answers and approximately 10x speed versus Responses. This does not establish an absolute latency target or accuracy parity with dlgo.
- OpenAI model documentation: https://developers.openai.com/api/docs/models/gpt-5.6-luna — pin actual availability/version and contract when implementing a provider adapter.

External references were located through web search; complete page contracts and live vendor performance were not independently tested in this review.

## Recommended sequence

1. Benchmark instrumentation, independent parity corpus and capacity accounting.
2. Cache parity/eviction coverage and cancellation/queue fixes.
3. Profiled fused SSM experiment, then chunked delta-rule prefill.
4. Cached-feature head ablations and audited data expansion in parallel.
5. Native 10-option coverage and true batched question suffixes.
6. Rerun paired vendor parity gates with pinned versions and cold/warm results.

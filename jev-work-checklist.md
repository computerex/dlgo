# dlgo System One / Jev-Compatible Inference Work Checklist

## Objective

Build a low-latency, low-cost, local **System One** inference mode for dlgo that works on supported small GGUF LLMs and is as close as practical to Jev's typed-decision API.

The primary contract is:

```text
state/context + typed question(s) -> typed answer(s) + normalized probabilities
```

The primary engine is **direct finite-choice likelihood scoring**, rather than ordinary autoregressive text generation:

1. Prefill shared state once.
2. Score every allowed typed answer continuation.
3. Normalize candidate sequence log-likelihoods into a probability distribution.
4. Return the selected typed value, all answer probabilities, timing, and diagnostics.

Grammar-constrained generation remains important for JSON/schema/tool calls, but it is not the primary fast-decision path.

---

## Product and compatibility contract

### Independent native decision-model work

**Non-reuse requirement:** dlgo's System One model, API implementation, prompt format, training corpus, training/evaluation pipeline, classifier weights, conversion tooling, and benchmarks must be independently created. Do not reuse Jeff/AutoJev code, weights, datasets, generated examples, templates, checkpoint metadata, or implementation details.

External projects may inform only broad independently-derived requirements: a native one-forward-pass readout is preferable to generated-token parsing; decision probabilities need held-out calibration; typed questions need isolated evaluation.

- [ ] Independently define a versioned System One prompt format and prompt-injection/data-boundary policy.
- [ ] Independently select and validate up to 255 single-token answer codes at the exact tokenizer answer boundary.
- [ ] Implement a newly initialized decision-readout tensor `[max_options, hidden_size]` for an untouched open base model.
- [x] Add a GGUF writer (`format/gguf/writer.go`) that can serialize an independently designed System One decision-readout tensor plus metadata as a standalone GGUF artifact (round-trip tested). `LoadSystemOneReadoutGGUF` reads it back with validation; `systemone-train -gguf-output` exports both formats in one run. Extending the existing model loader to read an embedded readout tensor remains pending.
- [x] Add a direct System One forward path: prompt prefill once, final hidden state, independently trained decision-readout matrix multiplication, mask to question option count, calibrated softmax. Implemented as `SystemOneHidden` (CPU `ForwardHidden`) + `ApplySystemOneReadout`; GPU-native variant in `gpu/pipeline.go` (`GpuPipeline.SystemOneHidden` batched prefill with `hiddenBuf` replacing the vocab projection).
- [x] Preserve generic continuation scoring strictly as a compatibility fallback for ordinary supported LLMs; document its weaker zero-shot quality and answer-code/position bias. Fallback remains for readout-less engines and wide questions; benchmark shows readout 85.0% vs fallback 80.0%.
- [x] Independently implement Jev-like question primitives: `choice` (keyed criteria), `noul` (boolean probability wrapper), and `score` (discretized numeric-range wrapper). JSON request/response shape remains pending.
- [x] Add `noul` and `score` to the HTTP endpoint and CLI surface: `/v1/systemone` accepts `type` (choice/noul/score) with `min`/`max` for score ranges; `dlgo systemone` CLI matches; both pass the trained readout when provided.
- [ ] Independently define answer semantics: choice probability map/choice/confidence, Noul probability, and score expectation/legend/confidence.
- [x] Add support for multiple independent questions per request as a batch of decision prompts, with correct per-question option masking. **Shared-prefill implemented**: `SystemOneMulti` prefills the shared prefix once; per-question suffixes run from cloned KV/SSM branches with token-boundary safety (BPE-unsafe questions fall back to standalone evaluation; exact E2E parity verified for readout and fallback engines). Benchmarks (3 questions, warm): 1.44× readout / 1.47× fallback speedup, growing with question count. Per-question graceful degradation (`Degraded`) for questions wider than the readout.
- [x] Train a newly initialized System One readout directly on final hidden states produced by the supplied frozen Q4_K_M GGUF backbone; packed Q4 backbone weights remain unchanged. Current best CPU artifact: `systemone-train/artifacts/qwen35-08b-q4-readout-v7-gp.json` — general-purpose corpus (20 domains, 6 zero-shot-only), zero-shot accuracy 86.1% on never-trained domains, overall test 81.4%/188, ECE 0.0585. Current best GPU artifact: `qwen35-08b-q4-readout-v8-gpu.json` (device=gpu, trained on Vulkan-extracted features) — zero-shot 81.75%, test 76.6%/188, ECE 0.0559; **readouts are device-aware** (`Device` field in JSON + GGUF metadata) and each engine refuses cross-device artifacts, because GPU and CPU extract structurally different hidden states.
- [x] Fit calibration temperature only using an independently held-out calibration split.
- [x] Add an on-disk feature cache (`-feature-cache`) to the trainer: extraction results are keyed by (ID, state, instructions, target, codes, model path, device, ctx) so repeat runs with new hyperparameters skip the expensive backbone passes. Cache auto-invalidates on backbone/dimension change; non-CPU devices add a device+hardware suffix so GPU features never collide with CPU entries.
- [x] Benchmark the independent decision model on this hardware against dlgo's generic fallback using identical inputs/settings. `cmd/systemone-bench` (20 test cases, Qwen3.5-0.8B Q4, ctx 512): fallback per-case 733ms (prefill 692ms + scoring 32ms, 80.0% acc) vs readout per-case 712ms (85.0% acc). Prefill dominates both paths; the readout removes the per-choice scoring cost and improves accuracy, but a shared-prefill/multi-question path is where the larger Jev-style latency win remains.

- [ ] Read the current official TypeSafe Jev API/reference documentation before locking dlgo's public request and response schemas.
- [ ] Record the Jev request fields, question types, answer types, response fields, probability semantics, and error behavior in this document or a companion compatibility specification.
- [ ] Define the compatibility target explicitly: exact wire/API compatibility where feasible, with documented dlgo extensions only where local-model constraints require them.
- [ ] Decide and document package ownership: a public `systemone` package, `dlgo` high-level wrappers, and lower-level `models/llm` execution primitives.
- [ ] Define stable typed question kinds for v1: boolean, single-choice enum, multi-choice enum, and optional numeric/range classification.
- [ ] Define typed answer representation that preserves stable application values separately from human-readable labels.
- [ ] Define handling for no-confidence / abstention / unsupported questions.
- [ ] Define concurrency guarantees: loaded pipeline reuse, request isolation, and grammar/decision state ownership.
- [ ] Define API versioning and model capability reporting.

### Proposed local Go API direction

```go
result, err := model.SystemOne(state, dlgo.SystemOneRequest{
    Questions: []dlgo.Question{
        {
            ID:   "route",
            Type: dlgo.QuestionChoice,
            Prompt: "Which queue should handle this request?",
            Choices: []dlgo.Choice{
                {Value: "billing", Label: "Billing issue"},
                {Value: "technical", Label: "Technical issue"},
                {Value: "account", Label: "Account access issue"},
            },
        },
    },
})
```

The final shape must be validated against Jev's official contract before it is published.

---

## Baseline investigation and reproducible benchmarks

- [ ] Add a benchmark harness dedicated to System One / decision inference.
- [x] Use `C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf` as the initial development and regression model.
- [x] Verify System One prompt construction bypasses automatic thinking priming; end-to-end model execution is in progress.
- [ ] Capture hardware, backend, thread count, context length, quantization, model identity, and dlgo revision in every benchmark record.
- [x] Build a small manually verified decision dataset covering boolean, three-way routing, moderation, and extraction-like choices.
- [x] Add adversarial cases: ambiguous input, out-of-domain input, choice-order permutations, and labels with different token lengths.
- [x] **General-purpose (domain-agnostic) corpus built** (`generate_gp_data.py` + `split_gp_data.py`): 20 domains with random 2–5 label subsets per example and 6 zero-shot-only domains; `systemone-eval` reports ZERO-SHOT vs SEEN-DOMAIN accuracy as the generalization gate. v7-gp passed: zero-shot 86.1% (137 rows) on domains never seen in training.
- [ ] Measure current unconstrained generation latency, token counts, and quality for every benchmark case.
- [ ] Measure current GBNF constrained-generation latency and quality for the same cases.
- [ ] Establish targets for prefill latency, per-question scoring latency, throughput, accuracy, and confidence quality.
- [ ] Create benchmark comparison reports for CPU and Vulkan GPU separately.

---

## Correctness repair: existing grammar path

- [ ] Add a regression test proving that the public `dlgo.LLM.Generate`, `Chat`, and `ChatMessages` APIs honor `WithGrammar`.
- [x] Fix `GenerateTextWithStopStrings` so it applies `GenerateConfig.Grammar` consistently.
- [ ] Avoid maintaining divergent decode loops: consolidate grammar-aware suppression, grammar masking, stop handling, streaming, and timing behavior.
- [ ] Ensure CPU pipeline, GPU pipeline, and server scheduler have consistent grammar behavior.
- [x] Ensure an incomplete grammar masks EOS and complete grammar permits EOS.
- [ ] Ensure grammar state is cloned or recreated per request; never mutate and reuse a caller's grammar instance unexpectedly.
- [x] Run focused grammar tests and relevant build/test suites after the repair. Full `go test ./...` remains blocked by documented pre-existing unrelated packages.

---

## Core direct candidate-scoring engine

### Execution primitive

- [x] Add a lower-level primitive that returns next-token logits immediately after a prompt prefill.
- [x] Add a non-mutating or snapshot-capable pipeline execution model suitable for scoring alternative continuations from the same prefixed state.
- [x] Inventory and correctly snapshot all branch-dependent state: KV cache, SSM/recurrent state, position, logits/run state, and any architecture-specific state.
- [x] Add safe, correct deep-copy branch state cloning for candidate branches; optimize/copy-on-write only after profiling.
- [ ] Ensure candidate scoring works correctly for standard attention, hybrid Qwen3.5 SSM/attention, and all currently supported tokenizers. Qwen3.8 hybrid end-to-end path now passes; standard-attention and tokenizer-family coverage remain pending.
- [x] Add a stable `logSoftmax` / `logSumExp` utility with numerical tests.
- [x] Score every candidate as the sum of conditional log probabilities of its continuation tokens.
- [x] Return raw sequence log probability, token count, optional mean token log probability, and normalized choice probability.
- [x] Define and implement explicit behavior for empty, duplicate, un-tokenizable, and too-long candidates.
- [x] Define behavior when candidate scoring exhausts available context.
- [x] Ensure scoring never mutates the shared prefilling state used by another candidate.

### Choice tokenization and prompting

- [x] Build a prompt compiler for System One questions.
- [x] Ensure Qwen3.8/Qwen3.5 decision prompts explicitly disable thinking; do not append the automatic `<think>\n` primer.
- [ ] Select a canonical answer prefix that produces reliable answer-label tokenization across supported models.
- [x] Use opaque, short answer codes in the scored continuation (for example `A`, `B`, `C`) while presenting semantic mappings in prompt context.
- [ ] Inspect and record candidate tokenizations for Qwen3.8 and representative supported tokenizer families.
- [ ] Ensure candidates do not accidentally tokenize with a preceding-space mismatch.
- [x] Test choice-order permutation behavior; stable value-derived automatic codes now prevent code reassignment, while semantic prompt-position bias remains observable and documented. **Root cause confirmed on held-out hand-crafted cases: both fallback and readout engines favor earlier answer codes (A > B > C) off-distribution. Fix implemented: `augment_data.py` generates code-permutation variants (shuffled explicit `codes` field) for training; trainer and eval accept explicit code assignments.**
- [ ] Test label-length bias and evaluate optional token-length normalization; keep raw sequence likelihood as the baseline unless evidence demonstrates otherwise.
- [ ] Add an explicit `other` / `unknown` choice option where callers require closed-set safety.

### Result construction

- [x] Implement deterministic top-choice selection with stable tie behavior.
- [x] Normalize all candidate scores into probabilities that sum to one.
- [x] Return answer ID/value, answer label, rank, raw score, probability, confidence, and timing metadata.
- [x] Define confidence as the selected answer's normalized probability for v1; document that this is not guaranteed calibrated confidence.
- [x] Add optional minimum-confidence abstention with a clear result/status.
- [ ] Add optional per-choice thresholds only after baseline calibration measurements justify them.
- [ ] Add result serialization tests.

---

## Jev-aligned multi-question API

- [x] Support a single state with multiple typed questions, mirroring Jev's principal efficiency model. `SystemOneMulti` evaluates a question list with per-question option masking; `/v1/systemone` accepts `type:"multi"` with a `questions` array (choice/noul/score per question). **Shared-prefill optimization complete** (see core section).
- [x] Reuse one state prefill across all questions when prompt semantics allow it. Implemented with token-boundary verification: tokenizing prefix and suffix separately must reproduce the concatenated prompt's token stream, else the question falls back to standalone prefill (results always identical to individual evaluation).
- [x] Decide whether questions are scored independently from the same state or encoded in one combined prompt; benchmark both approaches. Decision: independent per-question prompts (correct option masking, no cross-question leakage); combined-prompt mode deferred until benchmarked.
- [x] Preserve question IDs and answer types in result ordering.
- [x] Support boolean question answers without relying on generated `true` / `false` text.
- [x] Support single-choice enum questions.
- [x] Support numeric score questions via discretized integer buckets (2–64 wide), with graceful degradation when the range exceeds readout capacity.
- [ ] Determine whether multi-label/multiple-choice questions belong in v1 or require a separate independent-boolean scoring strategy.
- [x] Add a typed JSON representation modeled on the verified Jev request/response schema: `SystemOneRequest`/`SystemOneQuestion`/`SystemOneResponse` in `server/systemone.go` with `type` (choice/noul/score/multi), per-question `criteria`/`min`/`max`, and `min_confidence` abstention.
- [x] Add a dedicated local HTTP endpoint, initially proposed as `POST /v1/systemone`. Done and verified live (see server section).
- [ ] Add API documentation, examples, errors, and compatibility notes. Partial: `systemone-train/README.md` covers training/eval; request fixtures (`request-{score,noul,zeroshot-legal,zeroshot-medical,score-live,noul-live,multi-live,multi-live-gpu}.json`) document the wire shape; full API reference still pending.

---

## Fast constrained-generation fallback

- [ ] Retain general GBNF grammar for JSON, tool calls, schemas, and free-form constrained structures.
- [ ] Profile `Grammar.ApplyToLogits` on Qwen3.8 to quantify full-vocabulary scan, UTF-8 parsing, and allocation costs.
- [ ] Add a finite-literal-choice constraint representation separate from arbitrary GBNF.
- [ ] Compile finite allowed outputs into a tokenizer-aware trie/DFA once per model/tokenizer.
- [ ] At each trie state, mask only according to the small allowed next-token set instead of evaluating the full grammar over every vocabulary entry.
- [ ] Handle shared token prefixes, EOS, leading whitespace, and multi-token literals correctly.
- [ ] Make compiled constraints immutable and safe to share; keep decode state request-local.
- [ ] Add tests that compare trie-constrained output validity against equivalent GBNF grammars.
- [ ] Benchmark compiled-choice grammar against generic GBNF and direct candidate scoring.
- [ ] Use compiled-choice generation only when actual generated output is needed; direct scoring remains the default for typed decisions.

---

## CPU performance work

- [ ] Establish correct serial candidate scoring first.
- [ ] Measure branch-state copy cost separately from model forward cost.
- [ ] Add a candidate-scoring workspace to reuse allocations across requests.
- [ ] Eliminate per-token and per-candidate hot-path allocations where profiling confirms a benefit.
- [ ] Explore safe parallel scoring of independent candidates on CPU without oversubscribing BLAS worker pools.
- [ ] Benchmark 2, 3, 5, 10, and 50 choice cases.
- [ ] Evaluate hierarchical routing for large choice sets: coarse route followed by local reranking.
- [ ] Test short state, medium state, long state, and near-context-limit state behavior.

---

## Vulkan GPU performance work

- [x] Validate candidate scoring correctness on the current GPU pipeline. **System One GPU decision engine implemented** (`gpu/pipeline.go`): `GpuPipeline.SystemOneHidden` (batched prefill, vocab projection skipped, hidden downloaded from rs.X post-output-norm — mirrors CPU `ForwardHidden` semantics), `SystemOneWithReadout` (GPU extraction + CPU-side `ApplySystemOneReadout`), `SystemOneMulti` (per-question GPU prefill, wide questions degrade to CPU continuation scoring, per-result `Engine` attribution).
- [x] Ensure GPU scoring exposes/downloads logits only where necessary. The readout path downloads only the 1024-dim hidden state (no vocab projection, no logits); continuation-scoring fallback stays CPU-side.
- [x] Add CPU/GPU parity tests for scores, ranking, and probabilities within expected quantization tolerance. GPU E2E suite (`DLGO_SYSTEMONE_E2E=1`, `-tags "cgo vulkan"`): hidden parity (cosine ~0.97–0.98 vs CPU; probes proved the gap is structural — GPU matvec kernels quantize activations to Q8_1 — not KV/dp4a-switchable), device gates (cross-device readouts refused both directions), device metadata round-trip (JSON+GGUF), decision battery, GPU corpus accuracy gate (74.5–76.6% observed vs 72% gate; zero-shot 78.8% vs 76% gate), GPU multi with degradation.
- [x] Benchmark GPU latency across typical candidate cardinalities and prompt sizes. GPU question latency 135–235ms vs CPU 490–850ms (**3–4×**); GPU 4-question multi 1245ms (1 question degraded to CPU) vs CPU 3-question shared-prefill multi 1145ms; avg GPU corpus eval 145ms/question.
- [ ] Audit `GpuForwardBatchDecode` and dependent paths before relying on it for production branch batching; resolve incomplete/TODO behavior. (Still pending — unrelated to the readout path, which uses batched prefill.)
- [ ] Design a branched candidate scoring API with shared prefill and independent paged-KV block tables. (Deferred: GPU SSM state buffers are baked into per-layer GPU buffers; GPU multi already ~2× CPU multi, so branching isn't justified yet.)
- [ ] Batch candidates with the same continuation depth into GPU decode batches.
- [ ] Support variable-length candidate continuations with active-branch compaction.
- [ ] Minimize CPU↔GPU synchronization and vocabulary-logit downloads.
- [ ] Consider a GPU-side gather/log-softmax path for only the required candidate token IDs.
- [ ] Add CPU/GPU parity tests for scores, ranking, and probabilities within expected quantization tolerance.
- [ ] Benchmark GPU latency across typical candidate cardinalities and prompt sizes.

---

## Quality, confidence, and safety evaluation

- [ ] Build a labeled evaluation suite for target real-world System One tasks.
- [ ] Measure top-1 accuracy, macro-F1, per-class precision/recall, and abstention coverage.
- [x] Measure negative log likelihood, Brier score, expected calibration error (ECE), and reliability diagrams in `systemone-eval`. v4 readout on test: ECE 0.082; high-confidence bin [0.9-1.0) is near-perfectly calibrated (conf 0.970 vs acc 0.978, n=45); overconfidence concentrated in the 0.4-0.5 bin.
- [x] Test probability stability across seeds; direct scoring is deterministic without sampling.
- [x] Test sensitivity to choice ordering and candidate codes; prompt semantic-position bias was observed and remains a follow-up issue.
- [ ] Compare direct scoring against greedy constrained generation and ordinary prompted generation.
- [ ] Compare Qwen3.8 2B against other supported small models and quantizations.
- [ ] Document that typed constraints prevent invalid output forms, not incorrect decisions.
- [ ] Establish deployment thresholds for accept, abstain, and escalate outcomes.

---

## Server, CLI, and developer experience

- [x] Add `dlgo systemone` CLI command for local testing.
- [x] Add CLI support for state, questions, choices, JSON input/output, backend choice, and benchmark output. `dlgo systemone` accepts JSON input with `type` (choice/noul/score) and `min`/`max`; `cmd/systemone-bench` benchmarks fallback vs readout on JSONL cases.
- [x] Add `POST /v1/systemone` server route with validated typed request/response objects. Supports `type` (choice/noul/score/multi), per-question `criteria`/`min`/`max`, `min_confidence` abstention, a `degraded` flag when a request exceeds the readout's option capacity (falls back to continuation scoring instead of erroring), and an `engine` field ("gpu"/"cpu"). Requests run under the model's inference lock (mutually exclusive with generation). GPU-preferred when the model is GPU-resident and the readout is gpu-device; automatic CPU fallback on any GPU error or device mismatch. Verified live with the v7-gp general-purpose readout on zero-shot domains.
- [ ] Add an OpenAI-compatible extension only if it does not compromise Jev-style API compatibility.
- [ ] Expose backend, model, prompt token count, scoring token count, prefill time, scoring time, and total latency in optional diagnostics.
- [ ] Do not expose chain-of-thought or thinking text in System One results.
- [ ] Add Go API examples for boolean, routing, moderation, and multiple questions.
- [ ] Add Python/curl examples if the project server documentation supports them.
- [ ] Document model selection guidance and known small-model limitations.

---

## Testing and release gates

- [x] Unit-test numerical score accumulation and probability normalization.
- [x] Unit-test implemented typed question/answer validation rules.
- [x] Unit-test prompt compilation and code-to-value mapping.
- [x] Unit-test candidate state isolation; **concurrent request safety implemented**: `/v1/systemone` now runs under `Scheduler.WithInferenceLock`, mutually exclusive with the generation loop (shared KV/run-state/GPU command buffers made concurrent evaluation unsafe).
- [x] Unit-test Qwen3.5 hybrid SSM state cloning.
- [x] Add CPU integration tests using an available small local GGUF where test infrastructure permits. Opt-in Qwen3.8 hybrid end-to-end test passes (billing selected at 0.7060 probability for a duplicate-charge/refund case).
- [x] Add GPU integration/parity tests guarded by existing Vulkan build constraints. `gpu/systemone_gpu_test.go` (build tag `cgo && vulkan`, env-gated `DLGO_SYSTEMONE_E2E=1`): hidden parity, device gates, device metadata, decision battery, corpus accuracy gate, multi degradation — all passing on the RTX 4070 Ti SUPER.
- [x] Add regression tests for generic grammar behavior; existing grammar unit suite passes after the public-path repair.
- [ ] Add benchmark regression thresholds that flag material latency regressions.
- [ ] Run `go test ./...` after each implementation milestone.
- [ ] Produce a benchmark report comparing baseline generation, generic grammar, compiled-choice generation, and direct System One scoring.
- [ ] Review API compatibility against current Jev documentation immediately before initial public release.

---

## Proposed execution order

### Milestone 0 — Contract and baseline

- [ ] Verify Jev's official API contract and write the compatibility specification.
- [ ] Build the reproducible Qwen3.8 benchmark/evaluation harness.
- [x] Fix and test public CPU grammar routing.

### Milestone 1 — Correct System One core

- [ ] Implement serial direct candidate continuation scoring on CPU.
- [ ] Implement typed single-question boolean and enum API.
- [ ] Implement normalized probabilities, diagnostics, and minimum-confidence abstention.
- [ ] Validate correctness and quality on the initial benchmark set.

### Milestone 2 — Jev-style API surface

- [ ] Implement multiple typed questions over a shared state.
- [ ] Add high-level Go API, local CLI, and `POST /v1/systemone` endpoint.
- [ ] Add documentation and compatibility tests.

### Milestone 3 — Fast paths

- [ ] Implement tokenizer-aware compiled choice trie/DFA for constrained-generation fallback.
- [ ] Profile and optimize CPU state branching/workspaces.
- [ ] Implement and validate GPU candidate branch batching.

### Milestone 4 — Quality and production readiness

- [ ] Calibrate/evaluate confidence and abstention behavior.
- [ ] Compare supported small models and choose documented deployment defaults.
- [ ] Complete regression, load, concurrency, and latency testing.
- [ ] Publish benchmark and compatibility report.

---

## Decisions already made

- Direct finite-choice scoring is the primary System One mechanism.
- Ordinary token sampling is not used to choose System One answers.
- The returned answer must always be an allowed typed value; text parsing is not used to recover it.
- Raw likelihood scores are converted with log-softmax into a probability distribution over supplied allowed answers.
- Generic GBNF remains supported for structured generation, but is not the primary decision engine.
- A tokenizer-aware literal-choice trie/DFA is the planned grammar fast path.
- Qwen3.8/Qwen3.5 decision requests must explicitly disable thinking to protect latency.
- Initial development and evaluation use `C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf`.
- The public contract should track Jev closely, but verified official docs—not search-result summaries—must control field-level compatibility.

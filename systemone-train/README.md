# Local System One Decision-Model Training

## Goal

Produce an open, local, Jev-style decision model that is materially faster and more reliable than generic LLM continuation scoring:

```text
state + typed question + criteria
  -> one backbone forward pass
  -> learned option readout
  -> calibrated probability distribution
  -> typed answer
```

## Hardware and model constraints

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB VRAM.
- Development runtime: Python 3.12, PyTorch 2.6 CUDA 12.4.
- Available local generation model: `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`.
- GGUF is an inference quantization format and is **not** the training source. Fine-tuning must start from an FP16/BF16 Hugging Face checkpoint, then be exported to a dlgo-compatible decision GGUF artifact.

## Training strategy

### Non-reuse boundary

This project is built **from scratch**. Do not reuse Jeff/AutoJev code, checkpoints, decision heads, prompt templates, datasets, synthetic examples, training recipes, evaluation files, or conversion tooling.

Public projects may only be treated as high-level prior art for independently validating broad requirements such as: single-forward-pass decision readout, typed outputs, probability calibration, and held-out evaluation. Implementation, data, weights, prompts, and artifacts must originate in this repository and from independently sourced/permitted data.

### Stage A — build a native decision architecture

The initial model artifact is the locally supplied Q4_K_M GGUF:

```text
C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf
```

Training is deliberately **Q4-based**. We will not download or require an FP16/BF16 base checkpoint for the first milestone.

A quantized transformer cannot be conventionally full-weight fine-tuned in place: Q4 packed weight blocks have discrete codes/scales and ordinary gradient updates cannot safely alter them. The practical, independently built Q4 training design is therefore:

```text
frozen Q4 GGUF backbone
  -> dlgo forward pass to final normalized hidden state
  -> newly initialized FP32/BF16 System One readout
  -> cross-entropy training updates readout only
  -> fitted held-out temperature
```

This retains Q4 inference speed, uses little VRAM, and produces the key one-forward-pass decision path. The training head is entirely new and independently owned. Future Q4-compatible adapters may be added separately, but they are not required to establish a useful native decision model.

Design requirements:

- A newly initialized readout supports up to 255 answer slots.
- Prompt format is independently designed and versioned in this repository.
- Code vocabulary is selected from the tokenizer only after verifying each code is a unique, single token at the exact answer boundary.
- The readout is trained as a classifier, not derived from generated token likelihoods.
- The checkpoint stores only independently produced readout weights, calibration parameters, prompt version, tokenizer/model fingerprint, and dataset provenance.
- Head training uses dlgo's Q4 inference path as the frozen feature extractor; the model weights remain unmodified.

### Stage B — controlled initial training

Start from an untouched open base checkpoint and train our newly initialized decision head. On the 16 GB GPU, initial experiments should use:

1. Frozen backbone plus decision-head training to validate labels, prompt format, metrics, and export.
2. LoRA adaptation plus decision-head training, using BF16 and gradient accumulation.
3. Full-weight fine-tuning only after the preceding stages establish a measurable benefit and memory-safe reproducible recipe.

Use a hand-authored seed dataset first. Do not scale synthetic data until the data format, validators, and benchmark procedure are working.

### Stage C — independently generated training corpus

Construct a dataset around typed System One tasks:

- `choice`: routing, classification, moderation, selection, and grounded matching;
- `noul`: yes/no judgments;
- `score`: ordered qualitative scales;
- negative/instruction-injection controls;
- choice-order permutations;
- ambiguous/unknown cases;
- code and label wording variations.

Every family must have independently created train, development, calibration, and test partitions. Fit temperature exclusively on the calibration partition.

**General-purpose redesign (in progress):** a domain-agnostic corpus (`generate_gp_data.py`) with a 14-domain training taxonomy plus 6 zero-shot-only domains (medical urgency, legal, code review, finance, travel, education); every example draws a random 2–5 label subset so domain mappings cannot be memorized; permutation augmentation (`augment_data.py`, 2 variants) makes code letters meaningless; the acceptance gate is held-out accuracy on never-trained domains (`ZERO-SHOT` vs `SEEN-DOMAIN` breakdown in systemone-eval). First evidence the meta-task approach works: v6 (permutation-augmented) fixed a hand-crafted routing case the v5 readout got wrong with 0.96 overconfidence, collapsing confidence to a calibrated 0.56.

### Stage D — dlgo-native export and inference

Add native decision artifact support to dlgo:

- GGUF metadata: decision architecture/version, independently selected option codes/token IDs, calibration temperature, prompt layout version, and max option count.
- GGUF tensor: newly trained decision readout matrix `[max_options, hidden_size]`.
- Inference: retain final hidden state, multiply by decision readout, mask unavailable options, divide by fitted temperature, then softmax.
- API: Jev-like typed `/v1/systemone` semantics implemented independently.
- Keep generic candidate continuation scoring only as a fallback for ordinary GGUF LLMs.

## Data policy

- Never put API credentials in source code, git, JSONL data, model metadata, shell history, or logs.
- External teacher use is only for synthetic examples after a dataset schema, rate limits, spend cap, provenance record, and leakage checks are implemented.
- Store a provider key only in the process environment or a gitignored local secret configuration.
- Synthetic data must be kept distinct from held-out dev, calibration, and test sets.
- Every dataset row needs a task family, source/provenance, prompt, state, criteria, target, and split.

## Acceptance gates

1. **Correctness:** native decision-head probabilities sum to one, option masking is correct, no generated output is parsed.
2. **Compatibility:** request/response behavior matches the verified Jeff/Jev shape for supported primitives.
3. **Quality:** evaluate accuracy, macro-F1, NLL, Brier score, ECE, abstention coverage, and option-order sensitivity on held-out data.
4. **Performance:** measure end-to-end latency separately from prompt prefill and readout; compare to generic continuation scoring.
5. **Regression:** normal dlgo generation, grammar, CPU inference, and system-one fallback tests remain green.

## Immediate execution order

- [ ] Download and inspect Jeff-Qwen3.5-0.8B artifact and metadata.
- [ ] Implement a small native Python artifact inspector/evaluator to verify one-forward decision behavior locally.
- [ ] Define portable dlgo decision-artifact metadata and readout tensor names.
- [ ] Add dlgo model structures and a CPU native decision-readout inference path.
- [x] Create an initial hand-authored seed train corpus (15 rows) and held-out smoke evaluation corpus (6 rows); these validate plumbing only and are not sufficient for quality claims.
- [ ] Build teacher-data generation with strict secret handling and provenance, only after its schema and validation tests exist.
- [x] Train an initial readout-only Q4 artifact (`artifacts/qwen35-08b-q4-readout-v0.json`) and test native readout inference end-to-end in dlgo. Held-out generalization is currently poor, as expected for the tiny corpus.
- [ ] Add held-out calibration and native GGUF export for the independently trained artifact.

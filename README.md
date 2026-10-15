# dlgo

Fast LLM inference in pure Go. Run GGUF models locally on CPU or GPU — no Python, CUDA, or external inference runtime required.

## Quick Start

```bash
# Core chat CLI
go install github.com/computerex/dlgo/cmd/dlgo@latest

# Optional first-party commands
go install github.com/computerex/dlgo/cmd/dlgo-server@latest
go install github.com/computerex/dlgo/cmd/dlgo-vision@latest
go install github.com/computerex/dlgo/cmd/dlgo-image@latest
go install github.com/computerex/dlgo/cmd/mmproj-encode@latest

# Chat with a model (like ollama)
dlgo run model.gguf

# Start web server with UI
dlgo server --model model.gguf
```

`go install` puts commands in `$(go env GOPATH)/bin` (or `$(go env GOBIN)` when set); add that directory to your shell `PATH` before invoking them by name. `go install github.com/computerex/dlgo/cmd/...@latest` is also supported for portable CPU builds. GPU/Vulkan acceleration requires a source build with the `vulkan` tag and a cgo toolchain/Vulkan SDK; see [Building from Source](#building-from-source).

`gpu-watchdog` is intentionally Windows-only because it uses Windows Job Objects and PDH GPU performance counters. `test_glm` is a Vulkan diagnostic command, not a portable end-user CLI.

## Vision / multimodal inference

The repository includes a local multimodal path for an `mmproj` vision projector plus a compatible text GGUF backbone. `dlgo-vision` preprocesses an image, executes the projector, inserts the resulting image-embedding block at `<image>`, and generates directly from the combined sequence.

```bash
# Build (CPU path)
go build -ldflags "-linkmode internal" -o dlgo-vision ./cmd/dlgo-vision

# Describe an image. The projector output embedding width must equal the
# language model embedding width.
dlgo-vision \
  -mmproj C:\models\mmproj.gguf \
  -model C:\models\text-model.gguf \
  -image C:\images\example.png \
  -prompt "Describe this image in detail:\n<image>" \
  -max-tokens 128
```

`-image` also accepts deterministic synthetic fixtures (`white`, `black`, `gray`, `cb`, `red`, `green`, `blue`) for projector and image-token-placement checks. `cmd/mmproj-encode` is a companion utility for encoding/inspecting projector output. The vision stack lives in `vision/` and supports mmproj loading, CLIP-style preprocessing, patch/grid encoding, and projector output insertion into the language-model prompt.

The current implementation validates projector/text embedding-dimension compatibility at runtime and supports image grids needed by the embedding-generation path. It is an inference implementation, not a claim of compatibility with every vendor chat template or multimodal model family; validate the projector, text backbone, image prompt marker, and visual quality together for each model pairing.

## Diffusion image generation

In addition to language and vision-language inference, dlgo includes a local text-to-image diffusion pipeline. The current implementation targets **Z-Image / Lumina2-style DiT** checkpoints rather than claiming universal compatibility with every Stable Diffusion checkpoint family. It runs the complete pipeline locally:

```text
prompt -> Qwen text encoder -> quantized DiT denoising -> VAE decode -> PNG
```

The implementation includes a GGUF DiT loader, scheduler/sampler, 3D RoPE, quantized-weight CPU inference, Vulkan GPU DiT execution, SafeTensors VAE decoding, memory-budget estimation, and PNG output.

```bash
# CPU inference
go build -ldflags "-linkmode internal" -o dlgo-image ./cmd/dlgo-image

dlgo-image \
  --diffusion-model C:\models\z-image-turbo-Q4_K_M.gguf \
  --vae C:\models\ae.safetensors \
  --llm C:\models\qwen-text-encoder.gguf \
  -p "a cinematic photograph of a red fox in snow" \
  -o output.png -W 1024 -H 1024 --steps 8

# Vulkan-accelerated DiT inference
dlgo-image --gpu \
  --diffusion-model C:\models\z-image-turbo-Q4_K_M.gguf \
  --vae C:\models\ae.safetensors \
  --llm C:\models\qwen-text-encoder.gguf \
  -p "a cinematic photograph of a red fox in snow" -o output.png
```

`dlgo-image` validates dimensions and estimates peak memory before generation. The text encoder is released after prompt embedding so the DiT and VAE have the available memory budget. GPU acceleration applies to the DiT execution path; model compatibility, image quality, determinism, VRAM use, and throughput must be validated for each checkpoint/quantization/resolution combination. The diffusion implementation is in `models/diffusion/`, with CPU/GPU correctness and integration tests alongside it.

## CLI Usage

### `dlgo run` — Interactive Chat

Start an Ollama-style interactive chat session:

```
$ dlgo run qwen3.5-0.8b-q8_0.gguf --gpu

Loading qwen3.5-0.8b-q8_0.gguf (2.3s)

  Model:     qwen3.5
  Params:    24 layers, 1024 dim, 16 heads, vocab 151936
  Context:   8192 tokens
  Backend:   GPU (NVIDIA GeForce RTX 4070 Ti SUPER)
  Sampling:  temp=0.70 top-k=40 top-p=0.90

Type /help for commands, or start chatting.

>>> What is the capital of France?
The capital of France is Paris.

  45.2 tok/s | 32 tokens | 0.7s

>>> /help

  /help          Show this help
  /info          Show model info
  /clear         Clear conversation history
  /set temp N    Set temperature
  /set tokens N  Set max tokens
  /exit          Quit
```

**Run flags:**

| Flag | Default | Description |
|---|---|---|
| `--gpu` | false | Use Vulkan GPU backend |
| `--ctx N` | 8192 | Context length (tokens) |
| `--max-tokens N` | 512 | Max tokens per response |
| `--temp T` | 0.7 | Sampling temperature (0 = greedy) |
| `--top-k K` | 40 | Top-K sampling |
| `--top-p P` | 0.90 | Nucleus sampling |
| `--min-p P` | 0.0 | Min-P threshold |
| `--repeat-penalty R` | 1.1 | Repetition penalty |
| `--system "..."` | "You are a helpful assistant." | System prompt |
| `--threads N` | auto | Worker threads |
| `--no-stream` | false | Disable token streaming |

### `dlgo server` — Web UI & API

Start an HTTP server with a built-in chat web interface and an OpenAI-compatible API:

```
$ dlgo server --model model.gguf --gpu --port 8080

  dlgo server v0.1.0
  linux/amd64, 16 cores

  Web UI:  http://localhost:8080
  API:     http://localhost:8080/v1/chat/completions
  Health:  http://localhost:8080/health
```

The web UI lets you:
- Load and unload models dynamically
- Toggle between CPU and GPU backends
- Adjust temperature, top-p, top-k, max tokens
- Set system prompts
- Stream responses with live performance metrics

**Server flags:**

| Flag | Default | Description |
|---|---|---|
| `--model <path>` | | GGUF model to pre-load |
| `--gpu` | false | Use Vulkan GPU backend |
| `--host ADDR` | 0.0.0.0 | Bind address |
| `--port PORT` | 8080 | Listen port |
| `--ctx N` | 2048 | Context length |
| `--frontend <dir>` | auto-detect | Path to frontend dist/ |

**API endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming & non-streaming) |
| `/v1/systemone` | POST | Typed choice, noul, and score decisions |
| `/v1/models` | GET | List loaded models |
| `/v1/models` | POST | Load a model |
| `/v1/models` | DELETE | Unload a model |
| `/health` | GET | Health check |

## Go Library

Use dlgo as a Go package for embedding inference in your applications:

```go
model, err := dlgo.LoadLLM("model.gguf")
if err != nil {
    log.Fatal(err)
}

// Single-turn chat
response, _ := model.Chat("You are helpful.", "What is Go?")
fmt.Println(response)

// Streaming
model.ChatStream("", "Write a poem.", func(token string) {
    fmt.Print(token)
}, dlgo.WithMaxTokens(256))
```

**Options:** `WithMaxTokens(n)`, `WithTemperature(t)`, `WithTopK(k)`, `WithTopP(p)`, `WithSeed(s)`, `WithGreedy()`

## Building from Source

```bash
# CPU only (portable static binary with internal linking)
go build -ldflags "-linkmode internal" -o dlgo ./cmd/dlgo/

# With Vulkan GPU support
go build -tags vulkan -ldflags "-linkmode internal" -o dlgo ./cmd/dlgo/

# Build the web frontend (requires Node.js)
cd frontend && npm install && npm run build && cd ..

# Run server with frontend
dlgo server --model model.gguf --gpu --frontend frontend/dist
```

### Prerequisites

- **Go 1.21+**
- **Vulkan SDK** (optional, for GPU support) — install from [vulkan.lunarg.com](https://vulkan.lunarg.com/)
- **Node.js 18+** (optional, for building the web frontend)

## Features

- **25+ quantization formats** — Q4_0 through Q8_0, K-quants (Q2_K–Q8_K), I-quants (IQ1_S–IQ4_XS), MXFP4, F16, BF16, F32
- **Vulkan GPU inference** — full Vulkan compute backend with quantized MatVec shaders, fused attention, dp4a integer dot products
- **Never-OOM GPU** — automatic VRAM budget with graceful fallback to partial GPU + CPU
- **Mixture of Experts (MoE)** — fused multi-expert GPU dispatch, GPU-side top-K routing, zero CPU-GPU sync
- **Hybrid SSM+Attention** — Gated Delta Net recurrent layers (Qwen3.5, Qwen3-Coder-Next)
- **Multi-head Latent Attention (MLA)** — compressed KV cache (DeepSeek-V2, GLM-4.7)
- **Fast CPU** — AVX2/FMA/VNNI SIMD, parallel worker pools, batch prefill GEMM
- **Speech-to-text** — Whisper transcription
- **Voice activity detection** — Silero VAD

## Supported Architectures

| Architecture | Example Models | CPU tok/s | GPU tok/s |
|---|---|---:|---:|
| LLaMA | Llama 3.2 1B, TinyLlama 1.1B | 52–65 | 314–422 |
| Qwen2/3 | Qwen 2.5 0.5B, Qwen3 0.6B | 60–98 | 241–411 |
| Qwen3 MoE | Qwen3-Coder-30B-A3B (128 experts) | ~5.2 | ~40 |
| Qwen3.5 | Qwen3.5 0.8B–27B (hybrid GDN+attention) | 2.4–34 | 19–287 |
| Qwen3.5 MoE | Qwen3.5 35B-A3B, 122B-A10B | 1.4–4.1 | 2–11 |
| GLM-4.7 | GLM-4.7 Flash (MLA + MoE) | ~5.6 | ~15 |
| gpt-oss | gpt-oss-20b (MoE, attention sinks) | 4.5–5.6 | 33–52 |
| Gemma 2/3 | Gemma 3 270M–1B | 44–154 | 249–530 |
| SmolLM2 | SmolLM2 360M–1.7B | 42–96 | 177–411 |
| Phi | Phi-2, Phi-4-mini | 9–20 | ~125 |
| Whisper | Tiny, Base, Small | ~1x RT | — |

GPU benchmarks on NVIDIA RTX 4070 Ti SUPER. CPU with AVX2+FMA SIMD.

## Benchmarks vs Ollama

Same GGUF file, `temperature=0`, `seed=42`, `max_tokens=64`.

### GPU (Vulkan vs Ollama Vulkan)

| Model | Quant | dlgo | Ollama | Delta |
|---|---|---:|---:|---:|
| TinyLlama 1.1B | Q4_0 | 423 tok/s | 187 tok/s | **+126%** |
| Gemma 3 1B | Q4_K_M | 245 tok/s | 116 tok/s | **+111%** |
| Qwen 2.5 0.5B | Q4_K_M | 394 tok/s | 237 tok/s | **+66%** |

### GPU (Vulkan vs Ollama CUDA)

| Model | Quant | dlgo Vulkan | Ollama CUDA | Delta |
|---|---|---:|---:|---:|
| Qwen3.5 0.8B | Q8_0 | 287 tok/s | 250 tok/s | **+15%** |
| Qwen3.5 27B | Q3_K_M | 6.4 tok/s | 4.0 tok/s | **+60%** |

### CPU

| Model | Quant | dlgo | Ollama | Delta |
|---|---|---:|---:|---:|
| Qwen3.5 0.8B | Q8_0 | 33.2 tok/s | 27.3 tok/s | **+22%** |
| Qwen3.5 9B | Q3_K_M | 7.5 tok/s | 7.2 tok/s | **+4%** |
| Qwen3.5 27B | Q3_K_M | 2.5 tok/s | 2.4 tok/s | **+4%** |
| Qwen3.5 35B MoE | Q3_K_M | 8.1 tok/s | 7.6 tok/s | **+7%** |

## System One: Fast Typed Decisions

Alongside chat generation, dlgo has a local **System One** API for deterministic closed-set decisions: choice selection, binary verification (`noul`), and ordered scores. It is designed to mirror the typed request/response shape used by TypeSafe Jev while keeping inference local.

> **Scope and calibration:** System One head accuracy is only established for the evaluated distribution. A high binary probability is a preference between the two decision candidates, not a universal guarantee of truth. Evaluate and calibrate each application domain before automating consequential actions.

### API

`POST /v1/systemone` takes a loaded model, an on-disk readout artifact, a state, and one or more questions. The response contains typed answers and `Server-Timing` (`queue`, `decision`) measurements.

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-9b",
    "readout": "C:\\projects\\dlgo\\systemone-train\\artifacts\\qwen35-9b-q3-readout-v2-uni.json",
    "state": "A standard Pokemon type chart applies. Moltres is a Fire/Flying-type Pokemon.",
    "questions": {
      "moltres_water": {
        "type": "noul",
        "instructions": "Is Moltres, a Fire/Flying-type, weak to Water-type attacks?",
        "criteria": {
          "true": "Water attacks deal super-effective damage to Moltres.",
          "false": "Water attacks deal neutral, reduced, or no damage to Moltres."
        }
      }
    }
  }'
```

For `noul`, use semantic `true`/`false` criteria whenever possible. They are compiled as the candidate meanings while the public response remains the probability of `yes`:

```json
{"answers":{"moltres_water":{"type":"noul","noul":0.78}}}
```

This is materially more informative than a bare `yes`/`no` label for factual or policy verification.

### Architecture

The one-pass decision path is:

```text
state + question + semantic candidate criteria
        -> deterministic System One prompt compiler
        -> Qwen GGUF backbone prefill (CPU or Vulkan GPU)
        -> final normalized hidden state at "Answer code:"
        -> trained linear readout (one row per answer-code slot)
        -> normalized typed probabilities
```

- Prompts resolve candidate codes deterministically (`A`, `B`, …) by sorted value.
- A shared state prefix is prefilled once for multi-question requests; each question evaluates only its suffix.
- A readout supports up to five options; wider questions degrade rather than silently producing a misleading head result.
- A unified (`device: any`) head trains on both CPU and GPU feature snapshots so the same artifact can be served on either engine.
- `choice`, `noul`, and `score` are engine forms; training examples are closed-choice rows with explicit criteria and answer codes.

### Included unified Qwen heads

The three published unified artifacts below are versioned in `systemone-train/artifacts/` so a checkout can run the included heads immediately. Other experimental artifacts, feature caches, and reports remain local/gitignored. The evaluated `splits-gp2` results are:

| Backbone | Quant | Unified artifact | GPU test | CPU test | Notes |
|---|---|---|---:|---:|---|
| Qwen3.5 0.8B | Q4_K_M | `qwen35-08b-q4-readout-v10-uni` | 78.64% | 78.93% | Lowest-latency head |
| Qwen3.8 2B | Q4_K_M | `qwen38-2b-q4-readout-v2-uni` | 84.85% | 84.13% | Balanced head |
| Qwen3.5 9B | Q3_K_M | `qwen35-9b-q3-readout-v2-uni` | 93.65% | 93.51% | Highest corpus accuracy |

Each published head is available as both a human-inspectable JSON artifact and a compact GGUF export. Pass the JSON path as the API `readout` value; the GGUF export is provided for distribution and inspection tooling. The playground manifest references these versioned JSON artifacts by default.

The corpus metrics are not a broad factual-reasoning claim. `bench/systemone/compare_jev_suite.py` provides a separate, sealed cross-domain Jev comparison harness.

### Playground

Build the React frontend and launch a GPU playground server:

```powershell
cd frontend
npm install
npm run build
cd ..

# Fast 0.8B startup
.\bench\systemone\playground.ps1

# 9B decision model
.\bench\systemone\playground.ps1 -Size 9b
```

Open `http://localhost:8080`, choose **Playground**, select a model, edit state/questions, and inspect typed probabilities, latency, request history, and VRAM status. The Pokémon preset demonstrates semantic noul criteria; it is intentionally distinct from readout training examples.

For frontend HMR, run `npm run dev` in `frontend/`; Vite proxies System One requests to the server on port 8080.

### Train a unified head for another backbone

The System One trainer lives at `cmd/systemone-train`. A head must be trained from feature snapshots of the exact quantized backbone, context length, prompt format, and CPU/GPU engine configuration to be served.

1. Create JSONL splits with rows containing:
   - `id`, `state`, `instructions`, `criteria`, `codes`, and `target`;
   - at least two criteria per row;
   - explicit code assignments such as `{ "no": "A", "yes": "B" }` for robust GPU extraction.
2. Extract/cache features once per device. Re-running skips cache hits and checkpoints periodically.
3. Union-train from both caches to produce a `device: any` JSON readout and optional GGUF export.
4. Evaluate CPU and GPU separately, then validate against held-out, mechanism-disjoint examples.

Illustrative commands (paths and model IDs are local choices):

```powershell
# CPU feature cache fill
.\systemone-train.exe `
  -model C:\models\backbone.gguf `
  -train train.jsonl -dev dev.jsonl -calibration calibration.jsonl -test test.jsonl `
  -device cpu -feature-cache features-cpu.json `
  -output readout-cpu-tmp.json -epochs 1 -ctx 512

# GPU feature cache fill (run with no other GPU model process)
.\systemone-train-gpu.exe `
  -model C:\models\backbone.gguf `
  -train train.jsonl -dev dev.jsonl -calibration calibration.jsonl -test test.jsonl `
  -device gpu -feature-cache features-gpu.json `
  -output readout-gpu-tmp.json -epochs 1 -ctx 512

# Train one any-device artifact from the two complete caches
.\systemone-train-gpu.exe `
  -model C:\models\backbone.gguf `
  -train train.jsonl -dev dev.jsonl -calibration calibration.jsonl -test test.jsonl `
  -device cpu -feature-cache features-cpu.json -union-cache features-gpu.json `
  -output readout-uni.json -gguf-output readout-uni.gguf `
  -report readout-uni-report.json -epochs 100 -lr 0.02 -ctx 512
```

The included scripts in `bench/systemone/` are concrete examples for the 9B pipeline:

- `union-train-9b.ps1` / `union-train-9b-v3.ps1` / `union-train-9b-v4.ps1` — union training variants;
- `extract-9b-*.ps1` — feature-cache extraction;
- `validate-2b-uni.ps1`, `validate_v3.py` — validation helpers;
- `gen_aug2.py`, `gen_aug3.py` — experimental corpus augmentation generators;
- `compare_jev_pokemon.py`, `compare_jev_suite.py` — Jev 1.13 comparison harnesses.

### Data and evaluation

The current generated corpus and artifacts are intentionally ignored by Git:

```text
systemone-train/data/generated/
systemone-train/artifacts/
systemone-train/.env
```

The base `splits-gp2` data is structured state-to-choice supervision. It historically contained only `choice` rows, so applications using `noul` should supply semantic true/false criteria and must add independent noul validation. Keep evaluation entity/relation pairs out of training; do not promote a head using only random-split accuracy.

To compare a local 9B head to Jev 1.13, place an OpenRouter key only in ignored `systemone-train/.env`:

```text
OPENROUTER_API_KEY=sk-or-...
```

Then run:

```powershell
python .\bench\systemone\compare_jev_pokemon.py
python .\bench\systemone\compare_jev_suite.py
```

The suite sends the same state, question, and semantic criteria to Jev (`typesafe/jev-1.13`) and local dlgo. Result JSON is saved beneath the ignored `bench/results/` directory. Never commit API keys or raw secret-bearing requests.

### Latency snapshot

On the development workstation (Windows 11, RTX 4070 Ti SUPER 16 GB, Qwen3.5 9B Q3_K_M GPU resident), the 32-case semantic-noul comparison run measured:

| System | Mean end-to-end wall time | Accuracy on the 32-case suite | Mean Brier |
|---|---:|---:|---:|
| Jev 1.13 through OpenRouter | 282.8 ms | 31/32 (96.9%) | 0.0184 |
| dlgo Qwen3.5 9B v2 unified head, localhost | 345.1 ms | 30/32 (93.8%) | 0.0677 |

These are a development snapshot, not a service SLO or a benchmark claim. Jev timing includes network/provider execution; dlgo timing includes local HTTP and GPU inference. Use `Server-Timing` and your own workload/state lengths for p50/p95 measurements.


## Project Structure

```text
cmd/dlgo/              CLI entry point (run, server, info)
cmd/systemone-train/   System One feature extraction and head trainer
server/                HTTP server, model manager, scheduler
frontend/              React + Vite web UI and System One playground
models/llm/            LLM pipeline, prompt compiler, System One readout
systemone-train/       Local-only artifacts, caches, data, and secrets
bench/systemone/       System One launcher, corpus, validation, and comparison tools
gpu/                   Vulkan compute backend
format/gguf/           GGUF v2/v3 parser
quant/                 Quantization formats and SIMD dot products
blas/                  Matrix-vector multiply and worker pool
```

## License

MIT

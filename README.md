# dlgo

Fast LLM inference in pure Go. Run GGUF models locally on CPU or GPU — no Python, no CUDA, no dependencies.

## Quick Start

```bash
# Install
go install github.com/computerex/dlgo/cmd/dlgo@latest

# Chat with a model (like ollama)
dlgo run model.gguf

# Chat with GPU acceleration
dlgo run model.gguf --gpu

# Start web server with UI
dlgo server --model model.gguf --gpu
```

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
| `--top-p P` | 0.9 | Nucleus sampling |
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
| `/v1/models` | GET | List loaded models |
| `/v1/models` | POST | Load a model |
| `/v1/models` | DELETE | Unload a model |
| `/health` | GET | Health check |

**Example API call:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### `dlgo info` — Model Metadata

```
$ dlgo info model.gguf

File:           model.gguf
GGUF version:   3
Tensors:        291
Metadata keys:  26

general.architecture               qwen3
qwen3.context_length               32768
qwen3.embedding_length             1024
qwen3.block_count                  24
```

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

// Multi-turn conversation
response, _ = model.ChatMessages([]dlgo.Message{
    {Role: "system", Content: "You are a pirate."},
    {Role: "user", Content: "Tell me about the sea."},
    {Role: "assistant", Content: "Arrr, the sea be vast!"},
    {Role: "user", Content: "What about treasure?"},
}, dlgo.WithMaxTokens(128))
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
|---|---|---|---|
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
|---|---|---|---|---|
| TinyLlama 1.1B | Q4_0 | 423 tok/s | 187 tok/s | **+126%** |
| Gemma 3 1B | Q4_K_M | 245 tok/s | 116 tok/s | **+111%** |
| Qwen 2.5 0.5B | Q4_K_M | 394 tok/s | 237 tok/s | **+66%** |

### GPU (Vulkan vs Ollama CUDA)

| Model | Quant | dlgo Vulkan | Ollama CUDA | Delta |
|---|---|---|---|---|
| Qwen3.5 0.8B | Q8_0 | 287 tok/s | 250 tok/s | **+15%** |
| Qwen3.5 27B | Q3_K_M | 6.4 tok/s | 4.0 tok/s | **+60%** |

### CPU

| Model | Quant | dlgo | Ollama | Delta |
|---|---|---|---|---|
| Qwen3.5 0.8B | Q8_0 | 33.2 tok/s | 27.3 tok/s | **+22%** |
| Qwen3.5 9B | Q3_K_M | 7.5 tok/s | 7.2 tok/s | **+4%** |
| Qwen3.5 27B | Q3_K_M | 2.5 tok/s | 2.4 tok/s | **+4%** |
| Qwen3.5 35B MoE | Q3_K_M | 8.1 tok/s | 7.6 tok/s | **+7%** |

## Project Structure

```
cmd/dlgo/        CLI entry point (run, server, info)
cmd/dlgo-server/ Standalone HTTP server binary
server/          HTTP server, model manager, scheduler
frontend/        React + Vite web UI
dlgo.go          High-level Go API (LoadLLM, Chat, Generate)
models/llm/      LLM pipeline (tokenizer, forward, generation)
models/whisper/  Whisper speech-to-text
models/silero/   Silero VAD
gpu/             Vulkan compute backend
format/gguf/     GGUF v2/v3 parser
quant/           25+ quantization formats, SIMD dot products
blas/            Matrix-vector multiply, worker pool
ops/             Sampling, RoPE, norms, activations
memory/          KV cache, buffer pool
layers/          Conv1D, LSTM, GRU, MHA, GQA
audio/           WAV loading, mel spectrogram
examples/        Ready-to-run examples
bench/           Benchmark scripts and results
docs/            Additional documentation
```

## License

MIT

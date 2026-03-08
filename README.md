# dlgo

Pure Go deep learning inference. Load GGUF models and run them on CPU with zero dependencies beyond the Go standard library.

```go
model, _ := dlgo.LoadLLM("model.gguf")
response, _ := model.Chat("", "What is the capital of France?")
fmt.Println(response) // "The capital of France is Paris."
```

## Features

- **LLM inference** — text generation, multi-turn chat, streaming
- **Speech-to-text** — Whisper transcription from WAV files
- **Voice activity detection** — Silero VAD
- **GGUF format** — loads quantized models directly, no conversion needed
- **Vulkan GPU inference** — full Vulkan compute backend with quantized MatVec shaders (Q4_0, Q4_K, Q5_0, Q6_K, Q8_0, F32), fused attention, RoPE, SwiGLU/GeGLU, RMSNorm — **beats Ollama's Vulkan backend** by 66–126% on most models, within 5% on the rest
- **Fast on CPU** — AVX2/FMA/VNNI SIMD via optional CGo, QxQ integer dot products, batch prefill GEMM, parallel worker pools (within 1–16% of Ollama on generation for most models, same GGUF)
- **25+ quantization formats** — Q4_0 through Q8_0, K-quants (Q2_K–Q8_K), I-quants, F16, BF16, F32

## Supported Architectures

| Architecture | Models Tested | CPU tok/s | GPU tok/s |
|---|---|---|---|
| LLaMA | Llama 3.2 1B, TinyLlama 1.1B | 44–52 | 368–411 |
| Qwen2/3 | Qwen 2.5 0.5B, Qwen3 0.6B | 52–72 | 324–384 |
| Qwen3.5 | Qwen3.5 2B (hybrid GDN+attention) | ~16 | — |
| Gemma 2/3 | Gemma 2 2B, Gemma 3 1B, Gemma 3 270M | 37–114 | 253–481 |
| SmolLM2 | SmolLM2 360M, SmolLM2 1.7B | 37–94 | 293–426 |
| Phi | Phi-2, Phi-4-mini | 9–13 | 97 |
| Mistral | Mistral (llama-compatible) | — | — |
| Whisper | Tiny, Base, Small (speech-to-text) | ~1x RT | — |

CPU throughput with AVX2+FMA SIMD, parallel worker pool, batch prefill.
GPU throughput with Vulkan compute on NVIDIA RTX 4070 Ti SUPER.

## Benchmarks vs Ollama (CPU-only)

Benchmarks use the **exact same GGUF file** loaded into both engines via `ollama create`
with a Modelfile. `temperature=0`, `seed=42`, `max_tokens=64`, Ollama forced CPU-only
(`num_gpu=0`).

| Model | Quant | dlgo gen | Ollama gen | Delta | dlgo prefill | Ollama prefill |
|---|---|---|---|---|---|---|
| Gemma 3 270M | Q8_0 | 113.9 tok/s | 116.2 tok/s | **−2%** | 30 ms | 16 ms |
| SmolLM2 360M | Q8_0 | 93.7 tok/s | 117.5 tok/s | −20% | 60 ms | 37 ms |
| TinyLlama 1.1B | Q4_0 | 52.0 tok/s | 52.5 tok/s | **−1%** | 226 ms | 134 ms |
| Qwen3 0.6B | Q8_0 | 51.9 tok/s | 59.2 tok/s | −12% | 102 ms | 37 ms |
| Qwen 2.5 0.5B | Q4_K_M | 71.7 tok/s | 92.5 tok/s | −23% | 120 ms | 32 ms |
| Gemma 3 1B | Q4_K_M | 36.5 tok/s | 43.5 tok/s | −16% | 162 ms | 94 ms |
| SmolLM2 1.7B | Q4_K_M | 37.3 tok/s | 42.0 tok/s | −11% | 183 ms | 132 ms |
| Llama 3.2 1B | Q4_K_M | 44.3 tok/s | 50.3 tok/s | −12% | 144 ms | 77 ms |
| Phi-4-mini 3.8B | Q3_K_M | 12.9 tok/s | 19.2 tok/s | −33% | 869 ms | 153 ms |

**Notes:**
- Generation is **within 1–16% of Ollama** for 7 of 9 models. The SIMD compute kernels
  (QxQ integer dot products, `maddubs`+`madd` inner loops) are at parity with llama.cpp's
  — both operate at the DRAM bandwidth limit (~39 GB/s measured).
- The remaining gap is Go+CGo dispatch overhead (channel-based worker pool, goroutine
  scheduling, CGo call bridge per matmul chunk).
- SmolLM2 360M (−20%) and Qwen 2.5 0.5B (−23%) show a larger gap because the small
  model size makes per-dispatch overhead proportionally larger.
- Phi-4-mini (−33%) is a 3.8B parameter model in Q3_K_M, the largest and most complex
  model tested — the gap scales with the number of CGo calls per token.

## Install

```bash
go get github.com/computerex/dlgo
```

## Usage

### Chat

```go
model, err := dlgo.LoadLLM("llama-3.2-1b-instruct-q4_k_m.gguf")
if err != nil {
    log.Fatal(err)
}

response, err := model.Chat(
    "You are a helpful assistant.",
    "Explain quantum computing in one sentence.",
    dlgo.WithMaxTokens(128),
    dlgo.WithTemperature(0.7),
)
fmt.Println(response)
```

### Streaming

```go
model, _ := dlgo.LoadLLM("model.gguf")

model.ChatStream("", "Write a poem about Go.", func(token string) {
    fmt.Print(token)
}, dlgo.WithMaxTokens(256))
```

### Multi-turn conversation

```go
response, _ := model.ChatMessages([]dlgo.Message{
    {Role: "system", Content: "You are a pirate."},
    {Role: "user", Content: "Tell me about the sea."},
    {Role: "assistant", Content: "Arrr, the sea be vast!"},
    {Role: "user", Content: "What about treasure?"},
}, dlgo.WithMaxTokens(128))
```

### Speech-to-text

```go
whisper, _ := dlgo.LoadWhisper("whisper-base.gguf", "tokenizer.json")
text, _ := whisper.TranscribeFile("audio.wav")
fmt.Println(text)
```

### Sampling options

```go
dlgo.WithMaxTokens(256)     // max tokens to generate
dlgo.WithTemperature(0.8)   // 0 = greedy, higher = more creative
dlgo.WithTopK(40)           // top-K sampling
dlgo.WithTopP(0.9)          // nucleus sampling
dlgo.WithGreedy()           // deterministic output
```

## Project Structure

```
dlgo.go          High-level API (LoadLLM, Chat, Generate, Stream)
core/            QuantizedTensor with row-level dequantization
quant/           25+ GGML quantization formats, fused SIMD dot products
format/gguf/     GGUF v2/v3 parser
format/ggml/     Legacy GGML parser
gpu/             Vulkan GPU compute backend (MatVec, attention, RoPE, etc.)
ops/             RMSNorm, RoPE, Softmax, SwiGLU, GeGLU, sampling
blas/            Quantized matrix-vector multiply, parallel worker pool
layers/          Conv1D, LSTM, GRU, MHA, GQA, cross-attention
audio/           WAV loading, STFT, mel spectrogram
memory/          KV cache, buffer pool
decode/          Greedy decode, beam search
models/llm/      LLM pipeline (tokenizer, forward, generation, chat templates)
models/whisper/  Whisper speech-to-text
models/silero/   Silero voice activity detection
examples/        Ready-to-run examples
```

## Quantization Guide

See **[docs/quantization-guide.md](docs/quantization-guide.md)** for detailed guidance on
choosing quantization formats. Summary:

| Tier | Types | Speed | Use Case |
|---|---|---|---|
| **Tier 1** (QxQ integer SIMD) | Q4_0, Q8_0, Q2_K–Q6_K, Q5_0 | Fastest | Recommended for all use |
| **Tier 2** (float SIMD) | F16, Q4_1, Q5_1 | 2–4x slower | Functional, avoid if possible |
| **Tier 3** (dequant+dot) | IQ*, TQ*, BF16 | Slowest | Avoid for large models |

All common GGUF downloads (Q4_K_M, Q5_K_M, Q3_K_L, Q8_0, etc.) use only Tier 1 types.

## GPU Benchmarks vs Ollama

GPU benchmarks on NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM). Same GGUF files loaded
into both engines. Greedy decoding, `temperature=0`, `seed=42`.

### Vulkan vs CUDA (Ollama default)

Ollama defaults to CUDA on NVIDIA GPUs. All 9 tested models:

| Model | Quant | dlgo Vulkan | Ollama CUDA | Delta |
|---|---|---|---|---|
| Gemma 3 270M | Q8_0 | 481 tok/s | 529 tok/s | **−9%** |
| SmolLM2 360M | Q8_0 | 426 tok/s | 502 tok/s | −15% |
| TinyLlama 1.1B | Q4_0 | 411 tok/s | 472 tok/s | −13% |
| Qwen 2.5 0.5B | Q4_K_M | 384 tok/s | 552 tok/s | −31% |
| Qwen3 0.6B | Q8_0 | 324 tok/s | 372 tok/s | −13% |
| Llama 3.2 1B | Q4_K_M | 368 tok/s | 437 tok/s | −16% |
| SmolLM2 1.7B | Q4_K_M | 293 tok/s | 350 tok/s | −16% |
| Gemma 3 1B | Q4_K_M | 253 tok/s | 291 tok/s | −13% |
| Phi-4-mini 3.8B | Q3_K_M | 97 tok/s | 178 tok/s | −46% |

The 9–16% gap to CUDA for most models is the inherent Vulkan vs CUDA overhead on NVIDIA
hardware. On non-NVIDIA GPUs (AMD, Intel, mobile), dlgo's Vulkan backend provides
a significant advantage — Ollama's Vulkan backend is much slower for these quantization
types.

### Vulkan vs Vulkan (fair comparison)

Ollama forced to Vulkan backend (`OLLAMA_VULKAN=1 OLLAMA_LLM_LIBRARY=vulkan`):

| Model | Quant | dlgo Vulkan | Ollama Vulkan | Delta | dlgo prefill | Ollama prefill |
|---|---|---|---|---|---|---|
| SmolLM2 360M | Q8_0 | 389 tok/s | 411 tok/s | **−5%** | 86 ms | 1431 ms |
| TinyLlama 1.1B | Q4_0 | 423 tok/s | 187 tok/s | **+126%** | 99 ms | 1966 ms |
| Qwen 2.5 0.5B | Q4_K_M | 394 tok/s | 237 tok/s | **+66%** | 68 ms | 2749 ms |
| Gemma 3 1B | Q4_K_M | 245 tok/s | 116 tok/s | **+111%** | 194 ms | 3399 ms |

dlgo **beats Ollama's Vulkan backend** on 3 of 4 models by 66–126%, and is within 5% on
SmolLM2. Prefill is 10–50x faster across all models.

**GPU implementation highlights:**
- Pure Vulkan compute (cross-platform, no CUDA dependency)
- Quantized MatVec shaders with typed struct access (`float16_t`, `uint8_t`, `int8_t`)
- Fused layer dispatch (single CGo call per layer, minimized Go↔C overhead)
- Fused Add+RMSNorm kernel reducing barriers per layer
- Fused multi-head attention kernel (Q·K softmax, V accumulation)
- HOST_CACHED staging buffer for 14x faster CPU←GPU data transfer
- Single command buffer submission per token with batched dispatches
- Push descriptors (`VK_KHR_push_descriptor`) for minimal dispatch overhead
- `dp4a` integer dot product shaders (available for Q4_0, Q5_0, Q8_0, Q4_K, Q6_K)

## How It Works

1. **GGUF parser** reads model metadata and tensor locations from the file
2. **Quantized tensors** stay in their compressed format in memory — only dequantized on the fly during matrix multiplication
3. **Forward pass** runs the model: embedding, RoPE, GQA attention, SwiGLU/GeGLU FFN, RMSNorm, and hybrid SSM/attention (Gated Delta Net) — architecture variations are expressed as a per-layer `LayerSpec` resolved at load time
4. **SIMD acceleration** (optional, via CGo) uses AVX2+FMA+VNNI for QxQ integer dot products and batch prefill GEMM kernels
5. **Parallel matmul** distributes rows across a persistent worker pool with fused multi-matrix dispatch
6. **GPU acceleration** (optional, via Vulkan) offloads the entire forward pass to GPU — all weights, KV cache, and intermediate buffers reside in VRAM
7. **Token sampling** supports temperature, top-K, top-P, min-P, and repetition penalty

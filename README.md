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
- **Mixture of Experts (MoE)** — full MoE support with optimized expert-parallel dispatch (up to 512 experts), multiple gating functions (softmax, sigmoid, softmax-weight), shared experts
- **Hybrid SSM+Attention** — Gated Delta Net (GDN) recurrent layers with interleaved full attention, supporting Qwen3-Coder-Next, Qwen3.5, and Qwen3.5 MoE architectures
- **Multi-head Latent Attention (MLA)** — compressed KV cache with absorbed key projection for DeepSeek-V2 / GLM-4.7 architectures
- **Attention Sinks** — learned "trash bin" logits for OpenAI gpt-oss models, absorbing unneeded attention mass
- **Vulkan GPU inference** — full Vulkan compute backend with quantized MatVec shaders (Q4_0, Q4_K, Q5_0, Q6_K, Q8_0, F32), fused attention, RoPE, SwiGLU/GeGLU, RMSNorm, custom SSM/GDN kernels — **beats Ollama's Vulkan backend** by 66–126% on most models, within 5% on the rest; **beats Ollama CUDA on Qwen3.5** (+28%)
- **Fast on CPU** — AVX2/FMA/VNNI SIMD via optional CGo, QxQ integer dot products, fused IQ3_XXS/IQ4_XS SIMD kernels, batch prefill GEMM, parallel worker pools (**beats Ollama on Qwen3.5 models**, within 0–18% for most others)
- **25+ quantization formats** — Q4_0 through Q8_0, K-quants (Q2_K–Q8_K), I-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS), MXFP4, F16, BF16, F32

## Supported Architectures

| Architecture | Models Tested | CPU tok/s | GPU tok/s |
|---|---|---|---|
| LLaMA | Llama 3.2 1B, TinyLlama 1.1B | 52–59 | 421–441 |
| Qwen2/3 | Qwen 2.5 0.5B, Qwen3 0.6B | 59–91 | 367–447 |
| Qwen3 MoE | Qwen3-Coder-30B-A3B (128 experts, IQ3_XXS) | ~3.4 | — |
| Qwen3-Coder-Next | 80B-A3B (hybrid GDN+attention, 512 experts, IQ3_XXS) | ~5.8 | — |
| Qwen3.5 | Qwen3.5 0.8B (hybrid GDN+attention) | ~33 | ~287 |
| Qwen3.5 (large) | Qwen3.5 9B, 27B (hybrid GDN+attention) | 2.5–7.5 | 6.4–70.9 |
| Qwen3.5 MoE | Qwen3.5 35B-A3B (256 experts, 8 active) | ~8.1 | — |
| GLM-4.7 Flash | GLM-4.7 Flash (MLA + MoE, 64 experts, Q4_K) | ~5.1 | — |
| gpt-oss (OpenAI) | gpt-oss-20b (MoE, attention sinks, MXFP4/Q3_K_M) | 4.1–4.4 | — |
| Gemma 2/3 | Gemma 2 2B, Gemma 3 1B, Gemma 3 270M | 42–132 | 288–550 |
| SmolLM2 | SmolLM2 360M, SmolLM2 1.7B | 39–96 | 323–467 |
| Phi | Phi-2, Phi-4-mini | 9–19 | ~140 |
| Mistral | Mistral (llama-compatible) | — | — |
| Whisper | Tiny, Base, Small (speech-to-text) | ~1x RT | — |

CPU throughput with AVX2+FMA SIMD, parallel worker pool, batch prefill.
GPU throughput with Vulkan compute on NVIDIA RTX 4070 Ti SUPER.

## Benchmarks vs Ollama (CPU-only)

Benchmarks use the **exact same GGUF file** loaded into both engines via `ollama create`
with a Modelfile. `temperature=0`, `seed=42`, `max_tokens=64`, Ollama forced CPU-only
(`num_gpu=0`).

### Small models (fits in VRAM)

| Model | Quant | dlgo gen | Ollama gen | Delta | dlgo prefill | Ollama prefill |
|---|---|---|---|---|---|---|
| Gemma 3 270M | Q8_0 | 131.9 tok/s | 160.5 tok/s | −18% | 26 ms | 11 ms |
| SmolLM2 360M | Q8_0 | 96.1 tok/s | 117.7 tok/s | −18% | 59 ms | 37 ms |
| Qwen 2.5 0.5B | Q4_K_M | 90.8 tok/s | 121.0 tok/s | −25% | 66 ms | 24 ms |
| Qwen3 0.6B | Q8_0 | 58.7 tok/s | 69.9 tok/s | −16% | 69 ms | 33 ms |
| TinyLlama 1.1B | Q4_0 | 59.4 tok/s | 77.9 tok/s | −24% | 210 ms | 135 ms |
| Gemma 3 1B | Q4_K_M | 42.2 tok/s | 52.9 tok/s | −20% | 141 ms | 86 ms |
| Llama 3.2 1B | Q4_K_M | 52.4 tok/s | 58.6 tok/s | **−11%** | 108 ms | 51 ms |
| SmolLM2 1.7B | Q4_K_M | 39.2 tok/s | 44.8 tok/s | **−12%** | 160 ms | 124 ms |
| Qwen3.5 0.8B | Q8_0 | 33.2 tok/s | 27.3 tok/s | **+22%** | 191 ms | 85 ms |
| Phi-4-mini 3.8B | Q3_K_M | 19.4 tok/s | 23.2 tok/s | −16% | 288 ms | 137 ms |

### Large models (Qwen3.5 hybrid GDN+attention)

| Model | Quant | dlgo gen | Ollama gen | Delta | dlgo prefill | Ollama prefill |
|---|---|---|---|---|---|---|
| Qwen3.5 9B | Q3_K_M | 7.5 tok/s | 7.2 tok/s | **+4%** | 439 ms | 449 ms |
| Qwen3.5 27B | Q3_K_M | 2.5 tok/s | 2.4 tok/s | **+4%** | 1314 ms | 1528 ms |
| Qwen3.5 35B-A3B MoE | Q3_K_M | 8.1 tok/s | 7.6 tok/s | **+7%** | 640 ms | 441 ms |

### Frontier models (MoE, hybrid architectures, 20B+ params)

| Model | Quant | Size | Active Params | Architecture | CPU gen |
|---|---|---|---|---|---|
| Qwen3-Coder-Next 80B-A3B | IQ3_XXS | 27 GB | ~3B | Hybrid GDN+Attention, 512 experts | 5.8 tok/s |
| Qwen3-Coder-30B-A3B | IQ3_XXS | 12 GB | ~3B | MoE, 128 experts | 3.4 tok/s |
| gpt-oss-20b | MXFP4 | 11.5 GB | ~5B | MoE, 32 experts, attention sinks | 4.4 tok/s |
| gpt-oss-20b | Q3_K_M | 11 GB | ~5B | MoE, 32 experts, attention sinks | 4.1 tok/s |
| GLM-4.7-Flash | Q4_K_XL | 16.7 GB | ~3B | MLA + MoE, 64 experts | 5.1 tok/s |

**Notes:**
- **Qwen3.5 models outperform Ollama** on CPU generation: 0.8B (+22%), 9B (+4%), 27B (+4%), and 35B MoE (+7%).
  These use hybrid GDN+attention architectures where dlgo's optimized SSM kernels and fused IQ SIMD dot products give a measurable advantage.
- **Qwen3-Coder-Next** uses a novel hybrid architecture combining Gated Delta Net (GDN) SSM layers with full attention every 4th layer, plus MoE with 512 experts (10 active). Despite IQ3_XXS quantization, produces correct and coherent code output.
- **gpt-oss-20b** is an OpenAI MoE model with attention sinks (learned "trash bin" logits in softmax) and custom SwiGLU-OAI activation. Both MXFP4 (native) and Q3_K_M quantizations produce correct output.
- **GLM-4.7-Flash** uses Multi-head Latent Attention (MLA) with compressed KV cache and absorbed key projection, combined with MoE routing.
- Generation is within 11–25% of Ollama for most small models. The gap comes from Go+CGo
  dispatch overhead (channel-based worker pool, goroutine scheduling, CGo call bridge per matmul chunk).
- Phi-4-mini (−16%) is a 3.8B parameter model in Q3_K_M — gap scales with model size due to dispatch overhead.

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
| **Tier 1b** (fused SIMD) | IQ3_XXS, IQ4_XS | Fast | Optimized vpshufb/cvtepi8 kernels, used by MoE models |
| **Tier 2** (float SIMD) | F16, Q4_1, Q5_1 | 2–4x slower | Functional, avoid if possible |
| **Tier 3** (dequant+dot) | IQ1*, IQ2*, TQ*, BF16 | Slowest | Avoid for large models |

All common GGUF downloads (Q4_K_M, Q5_K_M, Q3_K_L, Q8_0, etc.) use only Tier 1 types.

## GPU Benchmarks vs Ollama

GPU benchmarks on NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM). Same GGUF files loaded
into both engines. Greedy decoding, `temperature=0`, `seed=42`.

### Vulkan vs CUDA (Ollama default)

Ollama defaults to CUDA on NVIDIA GPUs. All tested models:

| Model | Quant | dlgo Vulkan | Ollama CUDA | Delta |
|---|---|---|---|---|
| Qwen3.5 0.8B | Q8_0 | 287 tok/s | 250 tok/s | **+15%** |
| Qwen3.5 27B | Q3_K_M | 6.4 tok/s | 4.0 tok/s | **+60%** |
| Qwen3.5 9B | Q3_K_M | 70.9 tok/s | 86.7 tok/s | −18% |
| Gemma 3 270M | Q8_0 | 550 tok/s | 570 tok/s | **−3%** |
| SmolLM2 360M | Q8_0 | 467 tok/s | 545 tok/s | −14% |
| SmolLM2 1.7B | Q4_K_M | 323 tok/s | 403 tok/s | −20% |
| Qwen3 0.6B | Q8_0 | 367 tok/s | 421 tok/s | −13% |
| Llama 3.2 1B | Q4_K_M | 420 tok/s | 501 tok/s | −16% |
| Gemma 3 1B | Q4_K_M | 288 tok/s | 338 tok/s | −15% |
| Qwen 2.5 0.5B | Q4_K_M | 447 tok/s | 666 tok/s | −33% |
| TinyLlama 1.1B | Q4_0 | 441 tok/s | 623 tok/s | −29% |
| Phi-4-mini 3.8B | Q3_K_M | 140 tok/s | 201 tok/s | −31% |

Qwen3.5 is **28% faster** than Ollama's CUDA backend thanks to custom Vulkan compute
shaders for the Gated Delta Net (GDN) SSM layers — the first GPU-accelerated pure Vulkan
implementation of this architecture. The 7–25% gap to CUDA for standard attention models
is the inherent Vulkan vs CUDA overhead on NVIDIA hardware. On non-NVIDIA GPUs (AMD,
Intel, mobile), dlgo's Vulkan backend provides a significant advantage — Ollama's Vulkan
backend is much slower for these quantization types.

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
- Custom SSM/GDN shaders (conv1d+SiLU, delta rule, L2 norm, sigmoid gate) — full Gated Delta Net on GPU
- HOST_CACHED staging buffer for 14x faster CPU←GPU data transfer
- Single command buffer submission per token with batched dispatches
- Push descriptors (`VK_KHR_push_descriptor`) for minimal dispatch overhead
- `dp4a` integer dot product shaders (available for Q4_0, Q5_0, Q8_0, Q4_K, Q6_K)

## How It Works

1. **GGUF parser** reads model metadata and tensor locations from the file
2. **Quantized tensors** stay in their compressed format in memory — only dequantized on the fly during matrix multiplication
3. **Forward pass** runs the model: embedding, RoPE, GQA attention, Multi-head Latent Attention (MLA), SwiGLU/GeGLU FFN, RMSNorm, hybrid SSM/attention (Gated Delta Net with delta rule recurrence), attention sinks, and **Mixture-of-Experts** (MoE) with multiple gating functions — architecture variations are expressed as a per-layer `LayerSpec` resolved at load time
4. **SIMD acceleration** (optional, via CGo) uses AVX2+FMA+VNNI for QxQ integer dot products, fused vpshufb-based IQ3_XXS/IQ4_XS kernels, and batch prefill GEMM kernels
5. **Parallel matmul** distributes rows across a persistent worker pool with fused multi-matrix dispatch; MoE layers use batched multi-expert dispatch for full core utilization
6. **GPU acceleration** (optional, via Vulkan) offloads the entire forward pass to GPU — all weights, KV cache, and intermediate buffers reside in VRAM
7. **Token sampling** supports temperature, top-K, top-P, min-P, and repetition penalty

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
- **Fast on CPU** — AVX2/FMA/VNNI SIMD via optional CGo, QxQ integer dot products, batch prefill GEMM, parallel worker pools (within 13–20% of Ollama on generation)
- **25+ quantization formats** — Q4_0 through Q8_0, K-quants (Q2_K–Q8_K), I-quants, F16, BF16, F32

## Supported Architectures

| Architecture | Models Tested | Throughput |
|---|---|---|
| LLaMA | Llama 3.2 1B, TinyLlama 1.1B | ~60 tok/s |
| Qwen2/3 | Qwen 2.5 0.5B, Qwen3 0.6B | ~89 tok/s |
| Qwen3.5 | Qwen3.5 2B (hybrid GDN+attention) | ~16 tok/s |
| Gemma 2/3 | Gemma 2 2B, Gemma 3 1B, Gemma 3 270M | ~44–155 tok/s |
| SmolLM2 | SmolLM2 360M, SmolLM2 1.7B | ~99 tok/s |
| Phi | Phi-2, Phi-4-mini | ~8–9 tok/s |
| Mistral | Mistral (llama-compatible) | — |
| Whisper | Tiny, Base, Small (speech-to-text) | ~1x realtime |

Throughput measured with AVX2+FMA SIMD, parallel worker pool, batch prefill. CPU: AMD Ryzen 9 / Intel i9 class.

## Benchmarks vs Ollama (CPU-only)

All benchmarks use `temperature=0`, `seed=42`, `max_tokens=128`, Ollama forced CPU-only
with `num_gpu=0`. Three prompts averaged per model. Ollama prompt 1 is cold start; prompts
2–3 benefit from KV cache reuse.

| Model | Quant | dlgo gen | Ollama gen | Delta | dlgo prefill | Ollama prefill |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | Q4_0 | 62.5 tok/s | 74.7 tok/s | −16% | 171 ms | 119 ms |
| Qwen 2.5 0.5B | Q4_K_M | 91.9 tok/s | 115.2 tok/s | −20% | 68 ms | 43 ms |
| Gemma 3 1B | Q4_K_M | 44.2 tok/s | 50.9 tok/s | −13% | 196 ms | 83 ms |
| SmolLM2 360M | Q8_0 vs F16 | 99.3 tok/s | 62.1 tok/s | **+60%** | 53 ms | 29 ms |
| Gemma 3 270M | Q8_0 | 148 tok/s | — | — | 37 ms | — |

**Notes:**
- Generation throughput gap (13–20%) is due to Go+CGo overhead vs Ollama's native C++.
  The gap is consistent across models, meaning the SIMD kernels are at parity — the
  remaining cost is dispatch/scheduler overhead.
- SmolLM2 is faster because dlgo uses Q8_0 (integer SIMD) while Ollama serves F16
  (float SIMD with 4x fewer elements per instruction).
- Ollama prefill times for prompts 2–3 benefit from system prompt KV cache reuse.
  Cold-start prefill (prompt 1) is closer: TinyLlama dlgo=170ms vs Ollama=177ms.

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

## How It Works

1. **GGUF parser** reads model metadata and tensor locations from the file
2. **Quantized tensors** stay in their compressed format in memory — only dequantized on the fly during matrix multiplication
3. **Forward pass** runs the model: embedding, RoPE, GQA attention, SwiGLU/GeGLU FFN, RMSNorm, and hybrid SSM/attention (Gated Delta Net) — architecture variations are expressed as a per-layer `LayerSpec` resolved at load time
4. **SIMD acceleration** (optional, via CGo) uses AVX2+FMA+VNNI for QxQ integer dot products and batch prefill GEMM kernels
5. **Parallel matmul** distributes rows across a persistent worker pool with fused multi-matrix dispatch
6. **Token sampling** supports temperature, top-K, top-P, min-P, and repetition penalty

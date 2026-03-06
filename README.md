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
- **Fast on CPU** — AVX2/FMA SIMD via optional CGo, parallel matmul with worker pools
- **25+ quantization formats** — Q4_0 through Q8_0, K-quants (Q2_K–Q8_K), I-quants, F16, BF16, F32

## Supported Architectures

| Architecture | Models Tested | Throughput |
|---|---|---|
| LLaMA | Llama 3.2 1B, TinyLlama 1.1B | ~25 tok/s |
| Qwen2/3 | Qwen 2.5 0.5B, Qwen3 0.6B | ~30–40 tok/s |
| Gemma 2/3 | Gemma 2 2B, Gemma 3 1B, Gemma 3 270M | ~12–16 tok/s |
| SmolLM2 | SmolLM2 360M, SmolLM2 1.7B | ~12 tok/s |
| Phi | Phi-3/4 | — |
| Mistral | Mistral 7B | — |
| Whisper | Tiny, Base, Small | — |

Throughput measured on a single CPU core with Q4_K_M/Q8_0 quantization.

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
whisper, _ := dlgo.LoadWhisper("whisper-base.gguf")
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

## How It Works

1. **GGUF parser** reads model metadata and tensor locations from the file
2. **Quantized tensors** stay in their compressed format in memory — only dequantized on the fly during matrix multiplication
3. **Forward pass** runs the full transformer: embedding, RoPE, GQA attention, SwiGLU/GeGLU FFN, RMSNorm
4. **SIMD acceleration** (optional, via CGo) uses AVX2+FMA for fused dequant-dot-product kernels
5. **Parallel matmul** distributes rows across a persistent worker pool for large projections
6. **Token sampling** supports temperature, top-K, top-P, min-P, and repetition penalty

## License

MIT

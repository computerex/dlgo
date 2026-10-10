# Vision / image tokenization in dlgo

dlgo can encode an image into a block of token embeddings and splice that block
into a text model's prompt, producing image-conditioned generation on any model
whose embedding dimension matches the projector output.

## Pipeline

```
image bytes ─► preprocess ─► patch embed ─► ViT (CLIP) ─► projector ─► [][]float32
                                                                          │
                       prompt text + "<image>" marker ────────────────► splice
                                                                          │
                                             models/llm.GenerateDetailedWithEmbeddings
```

## Components

| File | Role |
|------|------|
| `vision/mmproj.go`  | Load a llama.cpp-format `mmproj` GGUF: config + weights (CLIP vision tower + LLaVA MLP projector). |
| `vision/preprocess.go` | Decode, resize (bilinear + antialias), normalize, extract patch features. |
| `vision/clip.go`    | Full CLIP forward: patch embed → class/pos → N blocks (bidirectional attention, GELU FFN) → post-LN → projector. |
| `models/llm/forward.go` | `ForwardEmbedding` — run a single externally-supplied embedding row through the LM. |
| `models/llm/generate.go` | `GenerateDetailedWithEmbeddings` — split the prompt on the image marker and interleave text segments with image rows. |
| `models/llm/tokenizer.go` | `EncodeNoBOS` — tokenize continuation segments without a second BOS. |
| `cmd/mmproj-encode` | Debug CLI: encode a synthetic image and dump embeddings (for diffing against `llama-mtmd-debug`). |
| `cmd/dlgo-vision`   | End-to-end CLI: mmproj + model + image → generation. |

## Supported projectors (v1)

`mlp` and `mlp_norm` (standard LLaVA-1.5 / LLaVA-1.6 MLP projectors), including
Yi `mm.1` variants. Other projector types (`qwen2vl`, `qwen25vl`, `qwen3vl`,
`gemma3`, `ldp`, …) are rejected with a clear error.

## Dimension matching

The projector output dimension must equal the target model's embedding length:

| Model | embedding_length |
|-------|------------------|
| Qwen3.5-0.8B | 1024 |
| Qwen3.6-35B-A3B / Qwen3.8-2B | 2048 |
| Qwen3.5-9B | 4096 |

A LLaVA-1.5-7B projector outputs **4096**, so it fits Qwen3.5-9B directly.

## Server (OpenAI-compatible multimodal input)

Start the server with a vision projector; image requests then go through the GPU path:

```sh
dlgo-server -gpu -ctx 8192 -model Qwen3.5-9B-Q3_K_M.gguf \
  -mmproj mmproj-llava15-7b-f16.gguf -port 8097
```

Send an OpenAI-style content array (the image part inserts an `<image>` marker):

```json
{"model":"...","messages":[{"role":"user","content":[
  {"type":"text","text":"Describe this image."},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
]}]}
```

`image_url.url` accepts a `data:` base64 URI, a local path / `file://`, or an
`http(s)://` URL. Without `-mmproj` (or on a CPU-only model) image requests
return a clear error.

## Performance

Real `mmproj-llava15-7b-f16.gguf` (623 MB, CLIP ViT-L/14, 576 image tokens × 4096):

| Stage | before | after |
|-------|-------:|------:|
| encoder (single image, CPU) | 21.9s | 5.0s |
| end-to-end 9B HTTP request (596 prompt tokens) | 34.3s | 14.9s |

The encoder speedup comes from batched SIMD-dot projections and token-parallel
forward. The remaining cost is the Qwen3.5 hybrid SSM recurrence, which is
inherently sequential per image token; reducing image-token count (e.g. 2×2
pixel-shuffle) is the next lever.

## Usage

Encode an image and dump embeddings:

```sh
go run ./cmd/mmproj-encode -mmproj mmproj-llava15-7b-f16.gguf -image cb -n 336 -compact
```

End-to-end image-conditioned generation:

```sh
go run ./cmd/dlgo-vision \
  -mmproj C:\Users\mohd\Downloads\mmproj-llava15-7b-f16.gguf \
  -model  C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf \
  -image cb -n 336 \
  -prompt "Describe this image in one sentence:\n<image>" \
  -max-tokens 48
```

`-image` accepts a PNG path or a synthetic pattern (`white|black|gray|cb`).

## Numerical validation

`vision` unit tests build a synthetic CLIP-shaped mmproj, load it, and compare
`Encoder.Encode` against an independent naive reference forward to
`relL2 ≤ 1e-4` (observed ≈ 1e-7) across class/pre-LN/layer-count configurations.

Real-file validation: `cmd/mmproj-encode` loads a 623 MB, 377-tensor
`openai/clip-vit-large-patch14-336` mmproj (f16) and produces 576 tokens ×
4096 dims with finite values.

## Notes / limitations

- Weights are dequantized to float32 at load (mmproj files are small).
- `spatial_unpad` (LLaVA-1.6) trims padded patch rows/cols and inserts
  `v.image_newline` embeddings between rows.
- No windowed attention / mRoPE VLMs (Qwen2-VL family) yet.
- The vision encoder itself runs on the CPU (now SIMD + multi-threaded); the
  LM forward runs on the GPU when the model is loaded with `-gpu`.
- A projector must match the target model's embedding space. A LLaVA-1.5-7B
  projector outputs LLaMA embeddings; on a Qwen model the image tokens are
  valid but not semantically meaningful without a Qwen-trained projector.

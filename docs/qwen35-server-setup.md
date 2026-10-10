# Qwen3.5-35B-A3B Local Server Setup

## Model

**File:** `C:\Users\mohd\Downloads\Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf`

## Patched Chat Template

The GGUF's built-in chat template has a bug: it wraps every historical assistant turn in an
empty `<think>...</think>` block even when `reasoning_content` is absent. This causes:

- Prompt drift across turns (hurts KV cache reuse)
- The model to emit tool call XML inside `<think>` blocks alongside native `tool_calls`
- Duplicate tool calls and token waste

**Patch file:** `C:\projects\dlgo\qwen35_chat_template.ginja`

The fix is a one-line change on line 99:

```jinja
{# Before: wraps EVERY historical assistant turn #}
{%- if loop.index0 > ns.last_query_index %}

{# After: only wraps when there is actual reasoning content #}
{%- if loop.index0 > ns.last_query_index and reasoning_content %}
```

## Launch Command

```powershell
cd C:\llama
.\llama-server.exe `
  -m "C:\Users\mohd\Downloads\Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf" `
  -c 131072 `
  -ngl 99 `
  --flash-attn on `
  --chat-template-file "C:\projects\dlgo\qwen35_chat_template.ginja"
```

### Flag reference

| Flag | Value | Purpose |
|------|-------|---------|
| `-m` | `C:\Users\mohd\Downloads\...Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf` | Model path |
| `-c` | `131072` | Context length (128K) |
| `-ngl` | `99` | GPU layers (all layers on GPU) |
| `--flash-attn on` | — | Flash attention (memory efficient) |
| `--chat-template-file` | `C:\projects\dlgo\qwen35_chat_template.ginja` | Patched Jinja2 template (fixes empty `<think>` wrapper bug) |

> **Important:** Use `--chat-template-file`, not `--chat-template`. The former accepts a file path;
> the latter treats the argument as a raw template string.

## Verify It's Working

```powershell
# Check server is running
curl http://127.0.0.1:8080/v1/models

# Quick tool-call test
curl -X POST http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf","messages":[{"role":"user","content":"hello"}],"max_tokens":20}'
```

The server exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

## Z-Harness Config (`~/.z.json`)

```json
{
  "api_url": "http://127.0.0.1:8080/v1/",
  "api_key": "not-needed",
  "model": "Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf"
}
```

## Known Issues

- The `reasoning_content` field in responses is the model's native thinking. The template fix
  prevents it from leaking into `full_content` when native `tool_calls` are present, but the
  reasoning itself is preserved in the API response field.
- Prompt caching works best when the template renders consistently across turns — the patch
  ensures tool-calling turns don't include empty `<think>` blocks that would break cache.

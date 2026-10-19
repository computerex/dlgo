$ErrorActionPreference = 'Stop'
$env:DLGO_SSM_BATCH = '1'
# Dense projection/FFN/SSM weights now run on the unified canonical format by
# DEFAULT: <=5-bit sources (e.g. this 9B's Q3_K) -> Affine4 (byte-lean decode),
# else Affine8 (coopmat prefill). Measured at 222 tokens: prefill ~4.3x faster
# AND decode ~1.2x faster, vs the native per-quant kernels. Set
# DLGO_UNIFIED_QUANT=0 to fall back to the legacy per-quant kernels.
$env:DLGO_UNIFIED_QUANT = '1'
& .\bench\systemone\server-9b.exe -host 127.0.0.1 -port 8097 -gpu -ctx 16384 -model 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf' -id qwen35-9b
if ($LASTEXITCODE -ne 0) { throw '9B server failed' }

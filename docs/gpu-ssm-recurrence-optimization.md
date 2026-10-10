# Safe fused GPU SSM recurrence

## Implementation

The forward paths now select the single-dispatch `ssm_delta_rule` recurrence for key/value head dimensions up to 128. This replaces decay, prediction, update and output (four dispatches, three internal barriers) without changing preprocessing, convolution, norm/gate, token order, CPU SSM paths or submission intervals.

The legacy fused shader was not equivalent to the decomposed reference: it read alpha as linear decay rather than applying `exp(alpha)` and used ordinary sums instead of the reference's compensated prediction/output sums. The repaired shader retains the reference loop order and `precise` intermediates, prevents multiply-add contraction in the update, and keeps each state column private to one invocation. C rejects dimensions outside its fixed array/workgroup limits; larger key dimensions automatically use the decomposed reference. The reference's existing value-column limit remains unchanged.

Set `$env:DLGO_SSM_REFERENCE='1'` **before starting the process** to force the four-dispatch GPU reference. The switch is read once at startup. Existing CPU SSM switches retain precedence.

## Rebuild and reproduce (PowerShell, repository root)

```powershell
& .\gpu\shaders\compile_ssm.ps1
spirv-val --target-env vulkan1.2 gpu/shaders/ssm_delta_rule.spv
go test -tags vulkan ./gpu -run '^TestSSMRecurrence' -count=3 -v
```

The focused compile script updates only the fused shader's embedded array, preserving unrelated shader binaries and registry entries. `compile.ps1` remains the full build source of truth and already includes this shader.

Numerical tests execute on a real GPU (fail, not skip, if unavailable): 256 changing-input recurrent steps, nonzero initial state, grouped keys, partial value columns, full 128-dimensional columns, output comparison every step, periodic/final state comparison, forced reference selection and automatic key-dimension fallback. Comparisons require identical float32 bits and reject nonfinite results.

## Measurements and scope

On an NVIDIA GeForce RTX 4070 Ti SUPER, the initial isolated backend run measured 512 recurrence steps at **25.934 ms reference / 10.996 ms fused (2.36x)**. Repeated isolated measurements ranged from **2.06x to 2.37x**, with absolute times varying from competing load. Each sample averages five alternating reference/fused runs after pipeline warmup and includes host recording, submission and GPU completion, but excludes input upload and downloads. This is not a GPU timestamp-only measurement or an end-to-end model prefill benchmark.

The final normal-package `-count=3` run passed all numerical tests and measured:

| Repeat | Reference | Fused | Speedup |
|---|---:|---:|---:|
| 1 | 26.085 ms | 11.237 ms | 2.32x |
| 2 | 25.226 ms | 10.643 ms | 2.37x |
| 3 | 25.426 ms | 10.749 ms | 2.37x |

Dispatch count is deterministically **2048 -> 512** for 512 recurrence steps: a **75% reduction in recurrence dispatches**. For 24 SSM layers this eliminates 36,864 recurrence dispatches/internal barriers per 512-token prefill. Including convolution, preprocessing and norm/gate, the per-token SSM compute sequence drops from seven to four dispatches (approximately 43%); copies, projections and attention remain.

Tested shapes produced bit-identical reference/fused outputs and sampled states on this GPU. This is not a claim of CPU float64 parity, model-logit parity, cross-vendor bit identity or universal numerical equivalence.

## Model E2E validation

Run after other GPU benchmarking completes (loads another model without stopping existing servers):

```powershell
$env:DLGO_SYSTEMONE_E2E='1'
go test -tags vulkan ./gpu -run '^TestSSMRecurrenceModelE2E$' -count=1 -v
```

The test uses Qwen3.5-0.8B Q4_K_M (24 layers, 18 SSM layers, 16 heads, 128x128 state), context 16,384, and alternates reference/fused runs on one pipeline. For its 10,944-token long prompt, fusion removes 590,976 recurrence dispatches and internal barriers (10,944 x 18 x 3). Each `SystemOneHidden` call resets model state and does not use prefix caching. Pipeline warmup is excluded. This measures single-question hidden extraction, not three-question HTTP/readout/restore latency and not first-ever pipeline startup.

Measured on RTX 4070 Ti SUPER, with the user's idle baseline server left loaded:

| Prompt | Tokens | Mean reference | Mean fused | Speedup | Latency reduction |
|---|---:|---:|---:|---:|---:|
| Short tenancy | 89 | 135.367 ms | 88.001 ms | 1.54x | 35.0% |
| Long synthetic GDPR | 10,944 | 16,470.755 ms | 10,896.136 ms | 1.51x | 33.8% |

All three alternating pairs for each prompt passed **bit-identical final hidden state** checks. Long-prompt pairs were 16,419.629/10,853.692 ms, 16,441.658/10,916.735 ms and 16,550.978/10,917.980 ms (reference/fused). This establishes tested model-hidden parity for these prompts/configuration, not universal model-logit or cross-vendor parity. Prefix-cache warm latency and three-question HTTP decisions were not measured here; the user owns those server benchmarks. Neither `pipeline.go` nor `systemone_state.go` was edited for this validation.

## Why not a multi-token shader yet?

A multi-token recurrence alone would need all convolved/normalized QKV and alpha/beta inputs materialized first. The current forward loop interleaves sequential convolution, preprocessing, recurrence and norm/gate with per-token scratch copies. Combining across tokens therefore requires a wider batch-buffer/preprocessing redesign or fusing convolution and normalization as well, with additional state layout and numerical validation. The existing repaired column-local recurrence provides measured gains without that risk. No parallel delta-rule reformulation or reordered reduction was introduced.

## Files

- `gpu/shaders/ssm_delta_rule.comp`, `.spv`: repaired fused math.
- `gpu/csrc/shaders_spirv.h`: only the corresponding embedded SPIR-V array regenerated.
- `gpu/csrc/vulkan_gpu.c`: fused dimension guards.
- `gpu/gpu_cgo.go`: wrapper contract documentation.
- `gpu/ssm_recurrence.go`: selection, environment switch and reference fallback.
- `gpu/forward.go`: four existing SSM forward sites select recurrence; existing user edits preserved.
- `gpu/ssm_recurrence_test.go`: focused numerical and latency tests.
- `gpu/ssm_recurrence_e2e_test.go`: opt-in cold-state model prefill/hidden parity test; `DLGO_SYSTEMONE_E2E=1`.
- `gpu/shaders/compile_ssm.ps1`: selective PowerShell rebuild.

The initial package test encountered an unused variable in concurrent `pipeline.go` work. An explicit-file backend build enabled validation without editing that file. A subsequent normal `./gpu` focused run built and passed once the working tree's package issue was resolved externally.

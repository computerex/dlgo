$ErrorActionPreference = 'Stop'
# Shader compiler was launched before this build sequence; wait without touching its script.
Wait-Process -Id 37604 -ErrorAction SilentlyContinue
$header = Get-Content gpu/csrc/shaders_spirv.h -Raw
if (-not $header.Contains('spv_ssm_recurrence_batch_t32')) { throw 'Shader regeneration incomplete' }
if ($header.Contains('spv_gemm_q3k_coopmat')) { throw 'Deleted coopmat pipeline still embedded' }
& powershell -NoProfile -ExecutionPolicy Bypass -File bench/systemone/test-latency-plan.ps1
if ($LASTEXITCODE -ne 0) { throw 'Test/build sequence failed' }
& python bench/systemone/run_latency_plan.py --profile-only
if ($LASTEXITCODE -ne 0) { throw 'Operator profile failed' }
& python bench/systemone/run_latency_plan.py --runs 30
if ($LASTEXITCODE -ne 0) { throw 'Matched T16/T32 benchmark failed' }
Write-Output 'LATENCY_PLAN_W1_AND_PROFILE_PASS'

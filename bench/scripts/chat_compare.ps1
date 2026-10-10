param([int]$MaxTokens = 4096)

$models = @(
    @{name="Qwen3.5-0.8B-Q8_0";    file="Qwen3.5-0.8B-Q8_0.gguf";            tokens=$MaxTokens}
    @{name="Qwen3-0.6B-Q8_0";      file="Qwen3-0.6B-Q8_0.gguf";              tokens=$MaxTokens}
    @{name="Qwen3.5-2B-Q4_K_M";    file="Qwen3.5-2B.Q4_K_M.gguf";            tokens=$MaxTokens}
    @{name="Llama-3.2-1B-Q4_K_M";  file="Llama-3.2-1B-Instruct-Q4_K_M.gguf"; tokens=$MaxTokens}
    @{name="Phi-4-mini-Q3_K_M";    file="Phi-4-mini-instruct-Q3_K_M.gguf";    tokens=$MaxTokens}
    @{name="gemma-3-1b-Q4_K_M";    file="gemma-3-1b-it-Q4_K_M.gguf";         tokens=$MaxTokens}
    @{name="gemma-2-2b-Q4_K_M";    file="gemma-2-2b-it-Q4_K_M.gguf";         tokens=$MaxTokens}
    @{name="Qwen3.5-9B-Q3_K_M";    file="Qwen3.5-9B-Q3_K_M.gguf";            tokens=$MaxTokens}
    @{name="gpt-oss-20b-Q3_K_M";   file="gpt-oss-20b-Q3_K_M.gguf";           tokens=2048}
)

$llamaExe = "C:\projects\llama.cpp\build\bin\Release\llama-cli.exe"
$dlgoDir = "C:\projects\dlgo"
$modelsDir = "C:\models"
$userPrompt = "Write a detailed essay about the history of artificial intelligence, from the earliest concepts through the modern era of deep learning and large language models."

Write-Host "=============================================================================="
Write-Host "  Chat Mode Comparison: dlgo vs llama.cpp (temp=0, thinking disabled)"
Write-Host "=============================================================================="

$summaryLines = @()

foreach ($m in $models) {
    $modelPath = Join-Path $modelsDir $m.file
    if (-not (Test-Path $modelPath)) {
        Write-Host "`n--- $($m.name): SKIPPED (not found) ---"
        continue
    }
    Write-Host "`n--- $($m.name) (target: $($m.tokens) tokens) ---"

    # --- DLGO ---
    Write-Host "  [dlgo] Running..." -NoNewline
    $dlgoOut = & go run -a -tags vulkan (Join-Path $dlgoDir "bench\scripts\raw_test_main.go") $modelPath $m.tokens chat_nothink 2>&1
    $dlgoStrs = @($dlgoOut | ForEach-Object { "$_" })
    $dlgoTps = "?"
    $dlgoTok = "?"
    foreach ($line in $dlgoStrs) {
        if ($line -match '(\d+\.\d+) tok/s.*?(\d+) tokens') {
            $dlgoTps = $Matches[1]
            $dlgoTok = $Matches[2]
        }
    }
    $dlgoTextLines = @()
    foreach ($line in $dlgoStrs) {
        if ($line -notmatch '^\[dlgo' -and $line -notmatch '^go ' -and $line -notmatch '^Using' -and $line -notmatch '^Prompt' -and $line -notmatch 'tok/s' -and $line -notmatch 'coopmat' -and $line -notmatch 'VK_KHR' -and $line -notmatch 'Memory heap' -and $line -notmatch 'Memory type' -and $line -notmatch 'vkCreate' -and $line -notmatch 'Initialized Vulkan' -and $line -notmatch 'Loaded.*pipelines') {
            $dlgoTextLines += $line
        }
    }
    $dlgoText = $dlgoTextLines -join " "
    $dlgoLen = $dlgoText.Length
    Write-Host " $dlgoTps tok/s, $dlgoTok tokens, ${dlgoLen} chars"
    if ($dlgoLen -gt 150) {
        Write-Host "  [dlgo] Start: $($dlgoText.Substring(0, 150))..."
        Write-Host "  [dlgo] End:   ...$($dlgoText.Substring($dlgoLen - 150))"
    } else {
        Write-Host "  [dlgo] Text: $dlgoText"
    }

    # --- LLAMA ---
    Write-Host "  [llama] Running..." -NoNewline
    $llamaOut = $userPrompt | & $llamaExe -m $modelPath -st -n $m.tokens --temp 0.0 -ngl 99 2>&1
    $llamaStrs = @($llamaOut | ForEach-Object { "$_" })
    $llamaTps = "?"
    $llamaTok = "?"
    foreach ($line in $llamaStrs) {
        if ($line -match 'eval time.*?(\d+\.\d+) tokens per second') {
            $llamaTps = $Matches[1]
        }
        if ($line -match 'eval time.*?/\s*(\d+) runs') {
            $llamaTok = $Matches[1]
        }
    }
    $llamaTextLines = @()
    foreach ($line in $llamaStrs) {
        if ($line -notmatch '^ggml_' -and $line -notmatch '^llama_' -and $line -notmatch '^load' -and $line -notmatch '^print_info' -and $line -notmatch '^common_' -and $line -notmatch '^sched_' -and $line -notmatch '^sampler' -and $line -notmatch '^generate' -and $line -notmatch '^system_info' -and $line -notmatch '^main:' -and $line -notmatch '^build' -and $line -notmatch 'available commands' -and $line -notmatch '/exit' -and $line -notmatch '/regen' -and $line -notmatch '/clear' -and $line -notmatch '/read' -and $line -notmatch 'model\s+:' -and $line -notmatch 'modalities' -and $line -notmatch 'Exiting' -and $line -notmatch '^>' -and $line.Trim().Length -gt 0) {
            $llamaTextLines += $line
        }
    }
    $llamaText = $llamaTextLines -join " "
    $llamaLen = $llamaText.Length
    Write-Host " $llamaTps tok/s, $llamaTok tokens, ${llamaLen} chars"
    if ($llamaLen -gt 150) {
        Write-Host "  [llama] Start: $($llamaText.Substring(0, 150))..."
        Write-Host "  [llama] End:   ...$($llamaText.Substring($llamaLen - 150))"
    } else {
        Write-Host "  [llama] Text: $llamaText"
    }

    $summaryLines += "$($m.name) | dlgo: $dlgoTps tok/s ($dlgoTok tok, ${dlgoLen}ch) | llama: $llamaTps tok/s ($llamaTok tok, ${llamaLen}ch)"
}

Write-Host "`n=============================================================================="
Write-Host "  SUMMARY"
Write-Host "=============================================================================="
foreach ($s in $summaryLines) { Write-Host "  $s" }
Write-Host "=============================================================================="
Write-Host "DONE"

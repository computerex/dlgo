$ErrorActionPreference = "Continue"

$models = @(
    @{name="gemma-3-1b-it"; path="C:\models\gemma-3-1b-it-Q4_K_M.gguf"; mode="chat"},
    @{name="Llama-3.2-1B"; path="C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"; mode="chat"},
    @{name="Phi-4-mini"; path="C:\models\Phi-4-mini-instruct-Q3_K_M.gguf"; mode="chat"},
    @{name="Qwen3-0.6B"; path="C:\models\Qwen3-0.6B-Q8_0.gguf"; mode="chat"},
    @{name="Qwen3.5-0.8B"; path="C:\models\Qwen3.5-0.8B-Q8_0.gguf"; mode="chat_nothink"},
    @{name="Qwen3.5-2B"; path="C:\models\Qwen3.5-2B.Q4_K_M.gguf"; mode="chat_nothink"},
    @{name="qwen2.5-0.5b"; path="C:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"; mode="chat"},
    @{name="smollm2-1.7b"; path="C:\models\smollm2-1.7b-instruct-q4_k_m.gguf"; mode="chat"},
    @{name="gemma-2-2b-it"; path="C:\models\gemma-2-2b-it-Q4_K_M.gguf"; mode="chat"},
    @{name="phi-2"; path="C:\models\phi-2.Q4_K_M.gguf"; mode="chat"},
    @{name="Qwen3.5-9B"; path="C:\models\Qwen3.5-9B-Q3_K_M.gguf"; mode="chat_nothink"}
)

$maxTokens = 500
$env:DLGO_GPU = "1"
$results = @()

foreach ($m in $models) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Testing: $($m.name)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $dlgoOut = go run -tags vulkan bench/scripts/raw_test_main.go $m.path $maxTokens $m.mode 2>&1
    $sw.Stop()
    
    $dlgoStdErr = ($dlgoOut | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }) -join "`n"
    $dlgoText = ($dlgoOut | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join " "
    
    $toks = 0
    if ($dlgoStdErr -match "(\d+\.?\d*)\s+tok/s.*?(\d+)\s+tokens") {
        $toks = [int]$Matches[2]
    }
    $tokps = 0
    if ($dlgoStdErr -match "(\d+\.?\d*)\s+tok/s") {
        $tokps = [float]$Matches[1]
    }
    
    $preview = if ($dlgoText.Length -gt 300) { $dlgoText.Substring(0, 300) + "..." } else { $dlgoText }
    
    $hasRepetition = $false
    $words = $dlgoText -split '\s+'
    if ($words.Count -gt 20) {
        $last20 = $words[-20..-1] -join " "
        $first100 = ($words[0..([math]::Min(99, $words.Count-1))]) -join " "
        $uniqueInLast20 = ($words[-20..-1] | Sort-Object -Unique).Count
        if ($uniqueInLast20 -lt 8) { $hasRepetition = $true }
    }
    
    $status = if ($hasRepetition) { "DEGENERATE" } else { "OK" }
    
    Write-Host "  Status: $status | $toks tokens | $tokps tok/s" -ForegroundColor $(if ($status -eq "OK") { "Green" } else { "Red" })
    Write-Host "  Preview: $preview"
    Write-Host ""
    
    $results += [PSCustomObject]@{
        Model = $m.name
        Status = $status
        Tokens = $toks
        TokPerSec = $tokps
        TimeMs = $sw.ElapsedMilliseconds
    }
}

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "SUMMARY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
$results | Format-Table -AutoSize

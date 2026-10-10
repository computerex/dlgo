$models = @(
    @{name="Qwen3.5-0.8B-Q8_0";         file="Qwen3.5-0.8B-Q8_0.gguf";         tokens=500}
    @{name="Qwen3-0.6B-Q8_0";           file="Qwen3-0.6B-Q8_0.gguf";           tokens=500}
    @{name="Qwen3.5-2B-Q4_K_M";         file="Qwen3.5-2B.Q4_K_M.gguf";         tokens=500}
    @{name="Llama-3.2-1B-Q4_K_M";       file="Llama-3.2-1B-Instruct-Q4_K_M.gguf"; tokens=500}
    @{name="Phi-4-mini-Q3_K_M";         file="Phi-4-mini-instruct-Q3_K_M.gguf"; tokens=500}
    @{name="gemma-3-1b-Q4_K_M";         file="gemma-3-1b-it-Q4_K_M.gguf";      tokens=500}
    @{name="gemma-2-2b-Q4_K_M";         file="gemma-2-2b-it-Q4_K_M.gguf";      tokens=500}
    @{name="Qwen3.5-9B-Q3_K_M";         file="Qwen3.5-9B-Q3_K_M.gguf";         tokens=300}
    @{name="gpt-oss-20b-Q3_K_M";        file="gpt-oss-20b-Q3_K_M.gguf";        tokens=300}
)

$llamaExe = "C:\projects\llama.cpp\build\bin\Release\llama-completion.exe"
$dlgoDir = "C:\projects\dlgo"
$modelsDir = "C:\models"

Write-Host "=" * 80
Write-Host "  dlgo vs llama.cpp Comparison Test"
Write-Host "=" * 80

foreach ($m in $models) {
    $modelPath = "$modelsDir\$($m.file)"
    if (-not (Test-Path $modelPath)) {
        Write-Host "`n--- $($m.name): SKIPPED (file not found) ---"
        continue
    }

    Write-Host "`n--- $($m.name) ($($m.tokens) tokens) ---"

    # Test with llama-completion (raw mode, no-cnv)
    Write-Host "  llama.cpp: " -NoNewline
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $llamaExe -m $modelPath -p "The history of artificial intelligence begins" -n $m.tokens --temp 0.0 -ngl 99 -no-cnv 2>&1
        $sw.Stop()
        $elapsed = $sw.ElapsedMilliseconds
        $perfLine = ($output | Select-String "eval time.*tokens per second") 
        if ($perfLine) {
            $match = [regex]::Match($perfLine.Line, '(\d+\.\d+) tokens per second')
            if ($match.Success) {
                $tps = $match.Groups[1].Value
                Write-Host "${tps} tok/s (${elapsed}ms)"
            } else {
                Write-Host "done (${elapsed}ms)"
            }
        } else {
            Write-Host "done (${elapsed}ms)"
        }
        # Extract output text (lines after the metadata)
        $textLines = $output | Where-Object { $_ -is [string] -and $_ -notmatch "^(ggml_|llama_|load|print_info|common_|sched_|sampler|generate|system_info|main:)" }
        $text = ($textLines -join " ").Trim()
        if ($text.Length -gt 200) { $text = $text.Substring(0, 200) + "..." }
        Write-Host "  llama text: $text"
    } catch {
        Write-Host "ERROR: $_"
    }

    # Test with dlgo
    Write-Host "  dlgo:      " -NoNewline
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & go run -a -tags vulkan "$dlgoDir\bench\scripts\raw_test_main.go" $modelPath $m.tokens raw 2>&1
        $sw.Stop()
        $elapsed = $sw.ElapsedMilliseconds
        $perfLine = ($output | Select-String "tok/s")
        if ($perfLine) {
            $match = [regex]::Match($perfLine.Line, '(\d+\.\d+) tok/s')
            if ($match.Success) {
                $tps = $match.Groups[1].Value
                Write-Host "${tps} tok/s (${elapsed}ms)"
            } else {
                Write-Host "done (${elapsed}ms)"
            }
        } else {
            Write-Host "done (${elapsed}ms)"
        }
        $textLines = $output | Where-Object { $_ -is [string] -and $_ -notmatch "^\[dlgo|^go |^Using|^Prompt|tok/s" }
        $text = ($textLines -join " ").Trim()
        if ($text.Length -gt 200) { $text = $text.Substring(0, 200) + "..." }
        Write-Host "  dlgo text:  $text"
    } catch {
        Write-Host "ERROR: $_"
    }

    Write-Host ""
}

Write-Host "`n" + ("=" * 80)
Write-Host "  Tests complete"
Write-Host "=" * 80

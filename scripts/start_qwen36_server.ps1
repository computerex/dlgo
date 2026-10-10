# Start Qwen3.6-35B-A3B server with patched chat template
# Run this from PowerShell to start the llama.cpp server
# Qwen3.6 may still have the empty <think> wrapper bug — apply the same fix.

$modelPath  = "C:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
$template   = "C:\projects\dlgo\qwen36_chat_template.ginja"
$logFile    = "$env:TEMP\llama_server_qwen36.log"
$serverExe  = "C:\llama\llama-server.exe"

Write-Host "Starting llama-server with Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf..." -ForegroundColor Cyan
Write-Host "  Model:   $modelPath"
Write-Host "  Template: $template"
Write-Host "  Log:     $logFile"
Write-Host "  Port:    8080"
Write-Host ""

# Kill any existing instance on port 8080
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 1

# Start server with patched template
& $serverExe `
  -m $modelPath `
  -c 131072 `
  -ngl 99 `
  --flash-attn on `
  --chat-template-file $template `
  *> $logFile

Write-Host "Server started (PID: $($pid)). Log: $logFile" -ForegroundColor Green

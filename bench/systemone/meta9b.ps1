$m = 'C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf'
if (-not (Test-Path $m)) { Write-Output 'FILE MISSING'; exit 1 }
$len = (Get-Item $m).Length
Write-Output ("FILE_GB=" + [math]::Round($len/1GB, 2))
.\gguf_info.exe $m -meta |
  Select-String -NotMatch 'Ġ' |
  Select-String -NotMatch 'merges' |
  Select-String -NotMatch 'token_type' |
  Select-String -NotMatch 'chat_template' |
  Select-String -NotMatch '^\s*$'

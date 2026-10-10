param([string]$Model)
.\gguf_info.exe $Model -meta |
  Select-String -NotMatch 'Ġ' |
  Select-String -NotMatch 'merges' |
  Select-String -NotMatch 'token_type' |
  Select-String -NotMatch '^\s*$'

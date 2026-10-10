Write-Output "=== Downloads GGUF ==="
Get-ChildItem C:\Users\mohd\Downloads -Filter *.gguf -ErrorAction SilentlyContinue |
  Select-Object Name, @{N='GB';E={[math]::Round($_.Length/1GB,2)}} | Format-Table -AutoSize
Write-Output "=== dlgo models dir GGUF ==="
Get-ChildItem C:\projects\dlgo\models -Recurse -Filter *.gguf -ErrorAction SilentlyContinue |
  Select-Object FullName, @{N='GB';E={[math]::Round($_.Length/1GB,2)}} | Format-Table -AutoSize
Write-Output "=== any *27* or *32* gguf anywhere in Downloads/dlgo ==="
Get-ChildItem C:\Users\mohd\Downloads, C:\projects\dlgo -Recurse -Include *27*.gguf,*32*.gguf -ErrorAction SilentlyContinue |
  Select-Object FullName, @{N='GB';E={[math]::Round($_.Length/1GB,2)}} | Format-Table -AutoSize

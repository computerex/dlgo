$c = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("TotalRAM_GB: " + [math]::Round($c.TotalPhysicalMemory/1GB,1))
Write-Output ("FreeRAM_GB: " + [math]::Round($os.FreePhysicalMemory/1MB,1))
Write-Output ("LogicalCPUs: " + $c.NumberOfLogicalProcessors)
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv

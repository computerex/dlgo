# kill_stale_terminals.ps1 — Kills orphaned PowerShell processes from VS Code.
# Keeps the current process alive. Safe to run from a VS Code terminal.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\kill_stale_terminals.ps1

$myPid = $PID
$before = (Get-Process powershell -ErrorAction SilentlyContinue | Measure-Object).Count
$stale = Get-Process powershell -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne $myPid }
$stale | Stop-Process -Force -ErrorAction SilentlyContinue
$after = (Get-Process powershell -ErrorAction SilentlyContinue | Measure-Object).Count
$killed = $before - $after
Write-Host "Killed $killed stale powershell processes ($before -> $after, kept PID $myPid)"

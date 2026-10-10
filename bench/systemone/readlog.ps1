param([string]$Path, [string]$Pattern)
$text = Get-Content -LiteralPath $Path -Raw
$lines = $text -split "`r?`n"
if ($Pattern) {
  $lines | Where-Object { $_ -match $Pattern }
} else {
  $lines
}

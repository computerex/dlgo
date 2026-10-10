# Rebuild only the fused recurrence, preserving all other embedded shaders.
# Invoke from any working directory: & .\gpu\shaders\compile_ssm.ps1
$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'ssm_delta_rule.comp'
$spv = Join-Path $PSScriptRoot 'ssm_delta_rule.spv'
$header = Join-Path $PSScriptRoot '..\csrc\shaders_spirv.h'
& glslc --target-env=vulkan1.2 -O $source -o $spv
if ($LASTEXITCODE -ne 0) { throw 'SSM shader compilation failed' }
$bytes = [IO.File]::ReadAllBytes($spv)
$sb = [Text.StringBuilder]::new()
[void]$sb.Append("static const uint32_t spv_ssm_delta_rule[] = {`n")
for ($i = 0; $i -lt $bytes.Length; $i += 4) {
    [void]$sb.Append(('    0x{0:x8},' -f [BitConverter]::ToUInt32($bytes, $i)))
    if ((($i / 4 + 1) % 8) -eq 0) { [void]$sb.Append("`n") }
    else { [void]$sb.Append(' ') }
}
[void]$sb.Append("`n};")
$text = [IO.File]::ReadAllText($header)
$pattern = '(?s)static const uint32_t spv_ssm_delta_rule\[\] = \{.*?\r?\n\};'
if ([regex]::Matches($text, $pattern).Count -ne 1) { throw 'Expected exactly one fused SSM array' }
$text = [regex]::Replace($text, $pattern, $sb.ToString())
[IO.File]::WriteAllText($header, $text, [Text.Encoding]::ASCII)
Write-Host "Updated only spv_ssm_delta_rule in $header ($($bytes.Length) bytes)"

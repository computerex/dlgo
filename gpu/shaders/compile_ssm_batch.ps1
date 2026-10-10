$ErrorActionPreference='Stop'
$header=Join-Path $PSScriptRoot '..\csrc\shaders_spirv.h'
$text=[IO.File]::ReadAllText($header)
$items=@(@{name='ssm_conv1d_batch';buffers=4;push=12},@{name='ssm_preprocess_batch';buffers=5;push=24},@{name='ssm_recurrence_batch';buffers=5;push=20},@{name='ssm_norm_gate_batch';buffers=3;push=12})
foreach($item in $items){
 $name=$item.name
 $spv=Join-Path $PSScriptRoot "$name.spv"
 & glslc --target-env=vulkan1.2 -O (Join-Path $PSScriptRoot "$name.comp") -o $spv
 if($LASTEXITCODE -ne 0){throw "Compile failed: $name"}
 & spirv-val --target-env vulkan1.2 $spv
 if($LASTEXITCODE -ne 0){throw "Validation failed: $name"}
 $bytes=[IO.File]::ReadAllBytes($spv)
 $sb=[Text.StringBuilder]::new()
 [void]$sb.Append("static const uint32_t spv_$name[] = {`n")
 for($i=0;$i -lt $bytes.Length;$i+=4){[void]$sb.Append(('    0x{0:x8},' -f [BitConverter]::ToUInt32($bytes,$i)));if((($i/4+1)%8)-eq 0){[void]$sb.Append("`n")}else{[void]$sb.Append(' ')}}
 [void]$sb.Append("`n};`nstatic const size_t spv_${name}_size = sizeof(spv_$name);`n")
 $pattern="(?s)static const uint32_t spv_$name\[\] = \{.*?static const size_t spv_${name}_size = sizeof\(spv_$name\);\r?\n"
 if([regex]::IsMatch($text,$pattern)){$text=[regex]::Replace($text,$pattern,$sb.ToString())}
 else{$text=$text.Replace('typedef struct {',$sb.ToString()+"`n"+'typedef struct {')}
 $entry='    {"'+$name+'", spv_'+$name+', spv_'+$name+'_size, '+$item.buffers+', '+$item.push+'},'
 if(-not $text.Contains($entry)){$text=$text.Replace("};`n`n#endif",$entry+"`n};`n`n#endif").Replace("};`r`n`r`n#endif",$entry+"`r`n};`r`n`r`n#endif")}
 Write-Host "Compiled and validated $name ($($bytes.Length) bytes)"
}
[IO.File]::WriteAllText($header,$text,[Text.Encoding]::ASCII)

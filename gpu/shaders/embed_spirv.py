"""Embed existing SPIR-V using compile.ps1's canonical shader order and registry.
Run compile.ps1 for a full shader rebuild; this supports surgical shader iterations.
"""
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parent
script = (ROOT / "compile.ps1").read_text(encoding="utf-8-sig")
names = re.findall(r'@\{name="([^"]+)";', script.split('$header = @"', 1)[0])
registry = script.split("static const ShaderInfo shader_registry[] = {", 1)[1].split("};", 1)[0]
registry_names = re.findall(r'\{"([^"]+)"', registry)
header_path = ROOT.parent / "csrc" / "shaders_spirv.h"
# Validate everything before replacing the generated file.
arrays = []
for name in names:
    data = (ROOT / (name + ".spv")).read_bytes()
    if len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
        raise ValueError(f"invalid SPIR-V: {name}")
    words = struct.unpack(f"<{len(data)//4}I", data)
    rows = ["    " + " ".join(f"0x{w:08x}," for w in words[i:i+8]) for i in range(0, len(words), 8)]
    arrays.append(f"static const uint32_t spv_{name}[] = {{\n" + "\n".join(rows) +
                  f"\n}};\nstatic const size_t spv_{name}_size = sizeof(spv_{name});\n")
if not set(registry_names).issubset(names):
    raise ValueError("registry references missing arrays")
enum_text = (ROOT.parent / "csrc" / "vulkan_gpu.h").read_text()
enum_region = enum_text.split("PIPE_COUNT", 1)[0].split("typedef enum", 1)[-1]
ids = re.findall(r'^\s*(PIPE_\w+)\s*(?:=\s*\d+)?\s*,', enum_region, re.M)
if len(ids) != len(registry_names):
    raise ValueError(f"enum/registry length mismatch {len(ids)}/{len(registry_names)}")
text = "// Auto-generated SPIR-V shader data. Do not edit.\n#ifndef DLGO_SHADERS_SPIRV_H\n#define DLGO_SHADERS_SPIRV_H\n#include <stdint.h>\n#include <stddef.h>\n\n"
text += "\n".join(arrays)
text += "\ntypedef struct {\n    const char* name;\n    const uint32_t* code;\n    size_t code_size;\n    int num_buffers;\n    int push_const_size;\n} ShaderInfo;\n\nstatic const ShaderInfo shader_registry[] = {" + registry + "};\n\n#endif // DLGO_SHADERS_SPIRV_H\n"
header_path.write_text(text, encoding="ascii")
(ROOT.parent / "csrc" / "vulkan_gpu.c").touch()
print(f"Embedded {len(names)} shaders, {len(registry_names)} pipelines")

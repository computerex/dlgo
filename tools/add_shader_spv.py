"""Inject a brand-new shader's SPIR-V array + registry row into gpu/csrc/shaders_spirv.h.

Usage: python tools/add_shader_spv.py <name> <n_buffers> <push_const_size> <PIPE_COMMENT>

Unlike patch_spv.py (which replaces an existing array), this appends the array
for a shader that does not yet exist in the header, and adds its registry row at
the end of shader_registry[] (matching the enum order in vulkan_gpu.h).

Formatting matches gpu/shaders/compile.ps1 exactly so future patch_spv.py runs
keep validating.
"""
import struct
import sys

HDR = r"C:\projects\dlgo\gpu\csrc\shaders_spirv.h"
SPVDIR = r"C:\projects\dlgo\gpu\shaders"


def fmt(words):
    out = []
    line = ""
    for i, w in enumerate(words):
        line += "    0x%08x," % w
        if (i + 1) % 8 == 0:
            out.append(line)
            line = ""
        else:
            line += " "
    if line:
        out.append(line)
    return out


def main():
    if len(sys.argv) < 5:
        print("usage: add_shader_spv.py <name> <n_buffers> <push_const_size> <PIPE_COMMENT>",
              file=sys.stderr)
        sys.exit(2)
    name = sys.argv[1]
    nbufs = int(sys.argv[2])
    pcs = int(sys.argv[3])
    comment = sys.argv[4]
    spv = "%s\\%s.spv" % (SPVDIR, name)

    with open(HDR, "rb") as f:
        text = f.read().decode("ascii")
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")

    for ln in lines:
        if ln.strip().startswith("static const uint32_t spv_%s[]" % name):
            print("array for %s already present" % name, file=sys.stderr)
            sys.exit(1)

    with open(spv, "rb") as f:
        raw = f.read()
    words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    body = fmt(words)

    # Insert the array + size just before the ShaderInfo typedef.
    decl_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("typedef struct {"):
            decl_idx = i
            break
    if decl_idx is None:
        print("ShaderInfo typedef not found", file=sys.stderr)
        sys.exit(1)

    arr = ["static const uint32_t spv_%s[] = {" % name] + body + [
        "};",
        "static const size_t spv_%s_size = sizeof(spv_%s);" % (name, name),
        "",
    ]
    lines = lines[:decl_idx] + arr + lines[decl_idx:]

    # Append the registry row before the closing brace of shader_registry[].
    reg_start = None
    for i, ln in enumerate(lines):
        if "shader_registry[] = {" in ln:
            reg_start = i
            break
    if reg_start is None:
        print("shader_registry not found", file=sys.stderr)
        sys.exit(1)
    close = None
    for i in range(reg_start, len(lines)):
        if lines[i].strip() == "};":
            close = i
            break
    if close is None:
        print("shader_registry end not found", file=sys.stderr)
        sys.exit(1)

    row = ('    {"%s", spv_%s, spv_%s_size, %d, %d}, // %s'
           % (name, name, name, nbufs, pcs, comment))
    lines = lines[:close] + [row] + lines[close:]

    with open(HDR, "wb") as f:
        f.write(eol.join(lines).encode("ascii"))
    print("added spv_%s: %d words" % (name, len(words)))


if __name__ == "__main__":
    main()

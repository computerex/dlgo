"""Replace the embedded SPIR-V array for one shader in gpu/csrc/shaders_spirv.h.

Usage: python tools/patch_spv.py <shader_name> [spv_path]

Recompiling the whole shader set would rewrite every array (compiler-version
dependent), so patch just the shader that changed. The formatter is validated by
re-formatting the existing array and requiring a byte-for-byte match first.
"""
import re
import struct
import sys

HDR = r"C:\projects\dlgo\gpu\csrc\shaders_spirv.h"


def fmt(words):
    # Must match gpu/shaders/compile.ps1 exactly: each word is "    0xXXXXXXXX,"
    # followed by a space, or a line break every 8 words.
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
    if len(sys.argv) < 2:
        print("usage: patch_spv.py <shader_name> [spv_path]", file=sys.stderr)
        sys.exit(2)
    name = sys.argv[1]
    spv = sys.argv[2] if len(sys.argv) > 2 else r"C:\projects\dlgo\gpu\shaders\%s.spv" % name

    with open(HDR, "rb") as f:
        text = f.read().decode("ascii")
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")

    decl = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("static const uint32_t spv_%s[]" % name):
            decl = i
            break
    if decl is None:
        print("array declaration not found for %s" % name, file=sys.stderr)
        sys.exit(1)

    body_start = decl + 1
    end = None
    for i in range(body_start, len(lines)):
        if lines[i].strip() == "};":
            end = i
            break
    if end is None:
        print("array end not found", file=sys.stderr)
        sys.exit(1)

    old_body = lines[body_start:end]
    old_words = [int(m, 16) for m in re.findall(r"0x([0-9a-fA-F]{8})", "".join(old_body))]

    if fmt(old_words) != old_body:
        print("FORMATTER MISMATCH — refusing to write", file=sys.stderr)
        sys.exit(1)
    print("formatter validated (%d words, %d lines)" % (len(old_words), len(old_body)))

    with open(spv, "rb") as f:
        raw = f.read()
    new_words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    new_body = fmt(new_words)

    new_lines = lines[:body_start] + new_body + lines[end:]
    with open(HDR, "wb") as f:
        f.write(eol.join(new_lines).encode("ascii"))
    print("patched spv_%s: %d words -> %d words" % (name, len(old_words), len(new_words)))


if __name__ == "__main__":
    main()

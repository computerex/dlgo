"""Replace the embedded SPIR-V array for one shader in gpu/csrc/shaders_spirv.h.

Recompiling the whole shader set would rewrite every array (compiler-version
dependent), so patch just the shader that changed. The formatter is validated by
re-formatting the existing array and requiring a byte-for-byte match first.
"""
import re
import struct
import sys

HDR = r"C:\projects\dlgo\gpu\csrc\shaders_spirv.h"
SPV = r"C:\projects\dlgo\gpu\shaders\attention_full_f32.spv"
NAME = "attention_full_f32"


def fmt(words):
    out = []
    for i in range(0, len(words), 8):
        chunk = words[i:i + 8]
        out.append("    " + ", ".join("0x%08x" % w for w in chunk) + ",")
    return out


with open(HDR, "rb") as f:
    text = f.read().decode("ascii")
# Preserve the file's existing line ending style (this file is CRLF on Windows).
eol = "\r\n" if "\r\n" in text else "\n"
lines = text.replace("\r\n", "\n").split("\n")

# Locate the array: declaration line, then body, then the closing "};"
decl = None
for i, ln in enumerate(lines):
    if ln.strip().startswith("static const uint32_t spv_%s[]" % NAME):
        decl = i
        break
if decl is None:
    print("array declaration not found", file=sys.stderr)
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

# Validate the formatter against the existing bytes.
if fmt(old_words) != old_body:
    print("FORMATTER MISMATCH — refusing to write", file=sys.stderr)
    print("expected first line:", repr(old_body[0]))
    print("produced first line:", repr(fmt(old_words)[0]))
    print("expected last line :", repr(old_body[-1]))
    print("produced last line :", repr(fmt(old_words)[-1]))
    sys.exit(1)
print("formatter validated against existing array (%d words, %d lines)" % (len(old_words), len(old_body)))

with open(SPV, "rb") as f:
    raw = f.read()
new_words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
new_body = fmt(new_words)

new_lines = lines[:body_start] + new_body + lines[end:]
with open(HDR, "wb") as f:
    f.write(eol.join(new_lines).encode("ascii"))
print("patched spv_%s: %d words -> %d words" % (NAME, len(old_words), len(new_words)))

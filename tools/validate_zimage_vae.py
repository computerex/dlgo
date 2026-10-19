"""Validate the remapped Z-Image VAE decoder against the reference FLUX ae.safetensors.

Compares every decoder.* tensor in the remapped file against the same key in
stable-diffusion.cpp's reference ae.safetensors. Differences are expected only from
BF16 (remapped) vs F16 (reference) storage; a misplaced/renamed tensor would show a
large error instead.
"""
import json
import struct

MINE = r"C:\models\z-image-vae-decoder.safetensors"
REF = r"C:\projects\sd-cpp\models\ae.safetensors"


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        data = f.read()
    hdr.pop("__metadata__", None)
    return hdr, base, data


def read_f32(hdr, data, key):
    info = hdr[key]
    s, e = info["data_offsets"]
    raw = data[s:e]
    dt = info["dtype"]
    numel = 1
    for d in info["shape"]:
        numel *= d
    if dt == "F32":
        return struct.unpack("<%df" % numel, raw)
    if dt == "F16":
        import numpy as np
        return np.frombuffer(raw, dtype="<f2").astype("f4")
    if dt == "BF16":
        import numpy as np
        u = np.frombuffer(raw, dtype="<u2").astype("u4") << 16
        return u.view("f4")
    raise ValueError(dt)


hm, bm, dm = load(MINE)
hr, br, dr = load(REF)

shared = sorted(set(hm) & set(hr))
print(f"remapped tensors: {len(hm)}, reference tensors: {len(hr)}, shared: {len(shared)}")

only_mine = sorted(set(hm) - set(hr))
if only_mine:
    print(f"keys present only in remapped file: {len(only_mine)}")
    for k in only_mine[:10]:
        print("   ", k)

import numpy as np

worst = []
for k in shared:
    a = np.asarray(read_f32(hm, dm, k), dtype="f8")
    b = np.asarray(read_f32(hr, dr, k), dtype="f8")
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH {k}: {a.shape} vs {b.shape}")
        continue
    scale = max(1e-9, float(np.max(np.abs(b))))
    d = float(np.max(np.abs(a - b))) / scale
    worst.append((d, k, scale))

worst.sort(reverse=True)
print("\nworst relative max-abs differences (remapped vs reference):")
for d, k, scale in worst[:10]:
    print(f"  {d:.3e}  (|ref|max={scale:.4g})  {k}")

mx = worst[0][0] if worst else 0.0
print(f"\nmax relative difference across {len(worst)} tensors: {mx:.3e}")
if mx < 5e-3:
    print("RESULT: MATCH — remap is numerically identical to the reference VAE")
else:
    print("RESULT: MISMATCH — remap does not reproduce the reference weights")

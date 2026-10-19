"""Remap Z-Image VAE safetensors (diffusers BF16 naming) to ldm/FLUX-style decoder keys.

Key mapping (diffusers -> ldm), with up-block index INVERTED (ldm counts up from
low-res; diffusers counts down from low-res):
  decoder.conv_in.*                              -> decoder.conv_in.*
  decoder.mid_block.resnets.0.*                  -> decoder.mid.block_1.*
  decoder.mid_block.resnets.1.*                  -> decoder.mid.block_2.*
  decoder.mid_block.attentions.0.group_norm.*    -> decoder.mid.attn_1.norm.*
  decoder.mid_block.attentions.0.to_q.*          -> decoder.mid.attn_1.q.*
  decoder.mid_block.attentions.0.to_k.*          -> decoder.mid.attn_1.k.*
  decoder.mid_block.attentions.0.to_v.*          -> decoder.mid.attn_1.v.*
  decoder.mid_block.attentions.0.to_out.0.*      -> decoder.mid.attn_1.proj_out.*
  decoder.up_blocks.{i}.resnets.{j}.norm1/conv1/norm2/conv2.*
                                                 -> decoder.up.{3-i}.block.{j}.*.*
  decoder.up_blocks.{i}.resnets.{j}.conv_shortcut.*
                                                 -> decoder.up.{3-i}.block.{j}.nin_shortcut.*
  decoder.up_blocks.{i}.upsamplers.0.conv.*      -> decoder.up.{3-i}.upsample.conv.*
  decoder.conv_norm_out.*                        -> decoder.norm_out.*
  decoder.conv_out.*                             -> decoder.conv_out.*
  encoder.*                                      -> dropped

Tensor bytes are copied verbatim (BF16); only the header (names/offsets) is rebuilt.
Linear to_q/k/v/out weights ([OC,IC]) are byte-identical to 1x1 conv weights ([OC,IC,1,1]).
"""
import json
import re
import struct
import sys

SRC = r"C:\Users\mohd\.cache\huggingface\hub\models--Tongyi-MAI--Z-Image-Turbo\snapshots\f332072aa78be7aecdf3ee76d5c247082da564a6\vae\diffusion_pytorch_model.safetensors"
DST = r"C:\models\z-image-vae-decoder.safetensors"


def map_key(k):
    if k.startswith("encoder."):
        return None
    if re.fullmatch(r"decoder\.conv_in\.(weight|bias)", k):
        return k
    if k.startswith("decoder.mid_block.resnets.0."):
        return "decoder.mid.block_1." + k.split("resnets.0.", 1)[1]
    if k.startswith("decoder.mid_block.resnets.1."):
        return "decoder.mid.block_2." + k.split("resnets.1.", 1)[1]
    if k.startswith("decoder.mid_block.attentions.0.group_norm."):
        return "decoder.mid.attn_1.norm." + k.split("group_norm.", 1)[1]
    if k.startswith("decoder.mid_block.attentions.0.to_q."):
        return "decoder.mid.attn_1.q." + k.split("to_q.", 1)[1]
    if k.startswith("decoder.mid_block.attentions.0.to_k."):
        return "decoder.mid.attn_1.k." + k.split("to_k.", 1)[1]
    if k.startswith("decoder.mid_block.attentions.0.to_v."):
        return "decoder.mid.attn_1.v." + k.split("to_v.", 1)[1]
    if k.startswith("decoder.mid_block.attentions.0.to_out.0."):
        return "decoder.mid.attn_1.proj_out." + k.split("to_out.0.", 1)[1]
    m = re.fullmatch(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(norm1|conv1|norm2|conv2|conv_shortcut)\.(weight|bias)", k)
    if m:
        i, j, name, wb = m.groups()
        if name == "conv_shortcut":
            name = "nin_shortcut"
        return f"decoder.up.{3 - int(i)}.block.{j}.{name}.{wb}"
    m = re.fullmatch(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(weight|bias)", k)
    if m:
        i, wb = m.groups()
        return f"decoder.up.{3 - int(i)}.upsample.conv.{wb}"
    if k == "decoder.conv_norm_out.weight":
        return "decoder.norm_out.weight"
    if k == "decoder.conv_norm_out.bias":
        return "decoder.norm_out.bias"
    if k.startswith("decoder.conv_out."):
        return k
    return None


with open(SRC, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))
    body = f.read()

header.pop("__metadata__", None)

# Build new header
new_header = {}
for k, info in header.items():
    nk = map_key(k)
    if nk is None:
        continue
    if nk in new_header:
        print(f"COLLISION: {nk}", file=sys.stderr)
        sys.exit(1)
    new_header[nk] = {"dtype": info["dtype"], "shape": info["shape"]}

# Sanity: verify channel flow matches dlgo expectations
def shape(nk):
    return new_header[nk]["shape"]

checks = [
    ("decoder.up.3.block.0.conv1.weight", [512, 512, 3, 3]),
    ("decoder.up.2.block.0.conv1.weight", [512, 512, 3, 3]),
    ("decoder.up.1.block.0.conv1.weight", [256, 512, 3, 3]),
    ("decoder.up.1.block.0.nin_shortcut.weight", [256, 512, 1, 1]),
    ("decoder.up.0.block.0.conv1.weight", [128, 256, 3, 3]),
    ("decoder.up.0.block.0.nin_shortcut.weight", [128, 256, 1, 1]),
    ("decoder.up.2.block.0.nin_shortcut.weight", None),  # must NOT exist
    ("decoder.up.3.block.0.nin_shortcut.weight", None),
]
ok = True
for nk, want in checks:
    has = nk in new_header
    if want is None:
        if has:
            print(f"CHECK FAIL: {nk} should not exist", file=sys.stderr)
            ok = False
    elif not has or list(shape(nk)) != want:
        print(f"CHECK FAIL: {nk} -> {shape(nk) if has else 'MISSING'}, want {want}", file=sys.stderr)
        ok = False
if not ok:
    sys.exit(1)
print("Channel-flow checks passed")

# Serialize with fresh contiguous offsets
out_header = {}
off = 0
blobs = []
for nk in sorted(new_header):
    src_key = next(k for k in header if map_key(k) == nk)
    info = header[src_key]
    s, e = info["data_offsets"]
    nbytes = e - s
    out_header[nk] = {
        "dtype": info["dtype"],
        "shape": info["shape"],
        "data_offsets": [off, off + nbytes],
    }
    blobs.append(body[s:e])
    off += nbytes

hb = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
hb += b" " * ((8 - (len(hb) % 8)) % 8)

with open(DST, "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    for b in blobs:
        f.write(b)

print(f"Wrote {DST}: {len(new_header)} tensors, {8 + len(hb) + off} bytes")

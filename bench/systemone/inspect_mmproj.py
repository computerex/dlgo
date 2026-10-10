import struct, sys

path = sys.argv[1]
f = open(path, 'rb')
assert f.read(4) == b'GGUF'
ver = struct.unpack('<I', f.read(4))[0]
nt = struct.unpack('<Q', f.read(8))[0]
nk = struct.unpack('<Q', f.read(8))[0]

def rdstr():
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

GT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def skipval(t):
    if t in GT:
        f.read(GT[t])
    elif t == 8:
        rdstr()
    elif t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            skipval(et)
    else:
        raise Exception('type %r' % t)

keys = []
for _ in range(nk):
    k = rdstr()
    t = struct.unpack('<I', f.read(4))[0]
    keys.append((k, t))
    skipval(t)

print('VERSION', ver, 'ntensors', nt, 'nkv', nk)
print('META:')
for k, t in keys:
    print(' ', k, 'type', t)

print('TENSORS (matching):')
for _ in range(nt):
    name = rdstr()
    nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
    tt = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    if any(x in name for x in ['post_ln', 'pre_ln', 'patch', 'position', 'class',
                               'blk.0.', 'mm.', 'image_newline', 'ln1', 'ln2',
                               'attn', 'ffn', 'norm', 'projector']):
        print(' ', name, dims, 'type', tt)

import base64, io, json, sys, time, urllib.request

# Build a synthetic checkerboard PNG (or use a file path passed as argv[1]).
import struct, zlib

def make_png(n=336):
    raw = bytearray()
    for y in range(n):
        raw.append(0)  # filter type 0
        for x in range(n):
            v = 255 if (x + y) % 2 == 0 else 0
            raw += bytes((v, v, v))
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        c += struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
        return c
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

png = make_png(336)
b64 = base64.b64encode(png).decode()
data_uri = "data:image/png;base64," + b64

prompt = "Describe this image. What pattern do you see?"

body = {
    "model": "Qwen3.5-9B-Q3_K_M",
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}
    ],
    "max_tokens": 48,
    "temperature": 0,
    "stream": False,
}

url = "http://127.0.0.1:8097/v1/chat/completions"
data = json.dumps(body).encode()
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read().decode())
    dt = time.time() - t0
    print("elapsed=%.1fs" % dt)
    print(json.dumps(out, indent=2)[:2000])
except urllib.error.HTTPError as e:
    dt = time.time() - t0
    print("elapsed=%.1fs HTTP %d" % (dt, e.code))
    print(e.read().decode())

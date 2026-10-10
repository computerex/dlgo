import json, time, urllib.request, base64, struct, zlib

def make_png(n, kind):
    raw = bytearray()
    for y in range(n):
        raw.append(0)
        for x in range(n):
            if kind == "cb":
                v = 255 if (x + y) % 2 == 0 else 0
            elif kind == "white":
                v = 255
            else:
                v = 0
            raw += bytes((v, v, v))
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        c += struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
        return c
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

def run(kind, text):
    b64 = base64.b64encode(make_png(336, kind)).decode()
    uri = "data:image/png;base64," + b64
    body = {
        "model": "Qwen3.5-9B-Q3_K_M",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}],
        "max_tokens": 200, "temperature": 0, "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:8097/v1/chat/completions",
                                 data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read().decode())
    dt = time.time() - t0
    msg = out["choices"][0]["message"]
    content = (msg.get("content") or "")
    # strip a leading thinking block for readability
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    return dt, out["usage"]["prompt_tokens"], content.strip()

text = "Answer with one word. Is this image mostly dark or mostly light?"
for kind in ("white", "black"):
    dt, pt, c = run(kind, text)
    print("=== %s (%.1fs, %d prompt tok) ===" % (kind, dt, pt))
    print(c[:300])
    print()

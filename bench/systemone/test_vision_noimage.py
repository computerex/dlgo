import json, time, urllib.request

def post(body):
    url = "http://127.0.0.1:8097/v1/chat/completions"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read().decode())
    return time.time() - t0, out

# No image: pure text baseline
body = {
    "model": "Qwen3.5-9B-Q3_K_M",
    "messages": [{"role": "user", "content": "Describe this image. What pattern do you see?"}],
    "max_tokens": 48, "temperature": 0, "stream": False,
}
dt, out = post(body)
msg = out["choices"][0]["message"]
print("NO-IMAGE elapsed=%.1fs prompt_tokens=%d" % (dt, out["usage"]["prompt_tokens"]))
print("content:", msg.get("content", "")[:400])

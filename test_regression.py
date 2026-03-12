import os, sys, json, time, requests, textwrap
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8080"
TIMEOUT = 600

SKIP_PREFIXES = ["ggml-vocab-", "parakeet-", "whisper-", "Kokoro_", "Nanonets-", "small_q8_0"]

def api(method, path, body=None):
    url = f"{BASE}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=TIMEOUT)
        elif method == "DELETE":
            r = requests.delete(url, json=body, timeout=TIMEOUT)
            return r.json() if r.text.strip() else {"status": "ok"}
        else:
            r = requests.post(url, json=body, timeout=TIMEOUT)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def unload_all():
    models = api("GET", "/v1/models")
    loaded = models.get("data", [])
    for m in loaded:
        resp = api("DELETE", "/v1/models", {"id": m["id"]})
        print(f"    Unloaded {m['id']}: {resp.get('status', resp)}")
    if loaded:
        time.sleep(3)

def load_model(mid, path):
    unload_all()
    resp = api("POST", "/v1/models", {"id": mid, "path": path, "gpu": True, "context": 2048})
    if resp.get("status") == "loaded":
        return True
    err_msg = resp.get("error", {}).get("message", "") if isinstance(resp.get("error"), dict) else str(resp.get("error", ""))
    if "already loaded" in err_msg:
        return True
    return False

def chat(model_id, messages, temp=0.0, max_tokens=512):
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.1,
    }
    t0 = time.time()
    resp = api("POST", "/v1/chat/completions", body)
    dt = time.time() - t0
    if "choices" not in resp:
        err = resp.get("error", "unknown error")
        if isinstance(err, dict):
            return None, 0, dt, err.get("message", str(resp))
        return None, 0, dt, str(err)
    text = resp["choices"][0].get("message", {}).get("content", "")
    tokens = resp.get("usage", {}).get("completion_tokens", 0)
    return text, tokens, dt, None

def wrap(text, width=100):
    lines = text.split("\n")
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width) or [""])
    return "\n".join(out)

def check_coherence(text, min_tokens=50):
    """Basic coherence checks."""
    issues = []
    if not text or not text.strip():
        issues.append("EMPTY output")
        return issues
    words = text.split()
    if len(words) >= 10:
        unique = len(set(w.lower() for w in words))
        ratio = unique / len(words)
        if ratio < 0.15:
            issues.append(f"REPETITIVE (unique ratio {ratio:.2f})")
    if len(text) > 200:
        chunks = [text[i:i+50] for i in range(0, len(text)-50, 25)]
        seen = set()
        for c in chunks:
            if c in seen:
                issues.append("REPEATED BLOCK detected")
                break
            seen.add(c)
    return issues

# ─── Discover models ───
all_models = api("GET", "/v1/models")
available = all_models.get("available", [])

chat_models = []
for m in available:
    mid = m["id"]
    skip = False
    for prefix in SKIP_PREFIXES:
        if mid.startswith(prefix):
            skip = True
            break
    if skip:
        continue
    chat_models.append(m)

# Also include already-loaded models not in available list
for m in all_models.get("data", []):
    if not any(c["id"] == m["id"] for c in chat_models):
        chat_models.append({"id": m["id"], "path": m.get("path", "")})

# Sort by file size (smallest first) for faster feedback
def get_size(m):
    try:
        return os.path.getsize(m.get("path", ""))
    except Exception:
        return float("inf")
chat_models.sort(key=get_size)

# Skip models whose file exceeds 80% of system RAM to prevent OOM crashes.
# On a 32 GB system this is ~25.6 GB. These models need more RAM than we have.
try:
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                     ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                     ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                     ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                     ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    MAX_MODEL_BYTES = int(stat.ullTotalPhys * 0.80)
    print(f"  System RAM: {stat.ullTotalPhys/(1<<30):.1f} GB, max model file: {MAX_MODEL_BYTES/(1<<30):.1f} GB")
except Exception:
    MAX_MODEL_BYTES = 25 * (1 << 30)  # 25 GB fallback

filtered = []
for m in chat_models:
    sz = get_size(m)
    if sz > MAX_MODEL_BYTES:
        print(f"  SKIPPING {m['id']} ({sz/(1<<30):.1f} GB > {MAX_MODEL_BYTES/(1<<30):.1f} GB limit)")
    else:
        filtered.append(m)
chat_models = filtered

print(f"{'='*80}")
print(f"  DLGO FULL REGRESSION TEST")
print(f"  Models to test: {len(chat_models)}")
print(f"  Tests per model: 4 (short, long, multi-turn, creative)")
print(f"{'='*80}\n")

results = {}

for idx, m in enumerate(chat_models):
    mid = m["id"]
    path = m.get("path", "")
    print(f"\n{'#'*80}")
    print(f"  [{idx+1}/{len(chat_models)}] {mid}")
    print(f"{'#'*80}")

    # Load
    print(f"  Loading (GPU)... ", end="", flush=True)
    if not load_model(mid, path):
        resp = api("POST", "/v1/models", {"id": mid, "path": path, "gpu": True, "context": 2048})
        err_msg = resp.get("error", {}).get("message", str(resp)) if isinstance(resp.get("error"), dict) else str(resp)
        print(f"FAILED: {err_msg}")
        results[mid] = {"status": "LOAD_FAILED", "error": err_msg}
        continue
    print("OK")

    model_results = {"status": "OK", "tests": {}}

    # ── Test 1: Short answer ──
    print(f"\n  [Test 1] Short answer (What is the capital of Japan?)")
    text, tok, dt, err = chat(mid, [
        {"role": "user", "content": "What is the capital of Japan? Answer in one sentence."}
    ], temp=0.0, max_tokens=64)
    if err:
        print(f"    ERROR: {err}")
        model_results["tests"]["short"] = {"status": "ERROR", "error": err}
    else:
        issues = check_coherence(text, min_tokens=3)
        status = "FAIL" if issues else "PASS"
        print(f"    [{status}] ({tok} tok, {dt:.1f}s): {text[:200]}")
        if issues:
            print(f"    Issues: {', '.join(issues)}")
        model_results["tests"]["short"] = {"status": status, "tokens": tok, "time": round(dt, 1), "text": text[:300], "issues": issues}

    # ── Test 2: Long generation (coherency test) ──
    print(f"\n  [Test 2] Long generation (~500 tokens, coherency test)")
    text, tok, dt, err = chat(mid, [
        {"role": "user", "content": "Write a detailed explanation of how the human immune system works, covering innate immunity, adaptive immunity, the role of T-cells and B-cells, and how vaccines help train the immune system. Be thorough and educational."}
    ], temp=0.0, max_tokens=700)
    if err:
        print(f"    ERROR: {err}")
        model_results["tests"]["long"] = {"status": "ERROR", "error": err}
    else:
        issues = check_coherence(text, min_tokens=100)
        status = "FAIL" if issues else "PASS"
        preview = wrap(text[:500])
        print(f"    [{status}] ({tok} tok, {dt:.1f}s)")
        for line in preview.split("\n")[:8]:
            print(f"    | {line}")
        if tok >= 400:
            print(f"    | ... ({tok} tokens total)")
        if issues:
            print(f"    Issues: {', '.join(issues)}")
        model_results["tests"]["long"] = {"status": status, "tokens": tok, "time": round(dt, 1), "text": text[:500], "issues": issues}

    # ── Test 3: Multi-turn conversation ──
    print(f"\n  [Test 3] Multi-turn (3 turns)")

    turns = [
        {"role": "user", "content": "I'm planning a trip to Paris. What are the top 3 things I should visit?"},
    ]
    text1, tok1, dt1, err1 = chat(mid, turns, temp=0.0, max_tokens=256)
    if err1:
        print(f"    Turn 1 ERROR: {err1}")
        model_results["tests"]["multi"] = {"status": "ERROR", "error": err1}
    else:
        print(f"    Turn 1 ({tok1} tok, {dt1:.1f}s): {text1[:150]}...")

        turns.append({"role": "assistant", "content": text1})
        turns.append({"role": "user", "content": "Great! Now tell me the best time of year to visit, and what kind of weather I should expect in each season."})

        text2, tok2, dt2, err2 = chat(mid, turns, temp=0.0, max_tokens=256)
        if err2:
            print(f"    Turn 2 ERROR: {err2}")
            model_results["tests"]["multi"] = {"status": "ERROR", "error": err2}
        else:
            print(f"    Turn 2 ({tok2} tok, {dt2:.1f}s): {text2[:150]}...")

            turns.append({"role": "assistant", "content": text2})
            turns.append({"role": "user", "content": "Based on everything you told me, create a 3-day itinerary for a spring trip to Paris."})

            text3, tok3, dt3, err3 = chat(mid, turns, temp=0.0, max_tokens=512)
            if err3:
                print(f"    Turn 3 ERROR: {err3}")
                model_results["tests"]["multi"] = {"status": "ERROR", "error": err3}
            else:
                issues = check_coherence(text3, min_tokens=50)
                status = "FAIL" if issues else "PASS"
                total_tok = tok1 + tok2 + tok3
                total_dt = dt1 + dt2 + dt3
                print(f"    Turn 3 ({tok3} tok, {dt3:.1f}s): {text3[:150]}...")
                print(f"    [{status}] Multi-turn total: {total_tok} tok, {total_dt:.1f}s")
                if issues:
                    print(f"    Issues: {', '.join(issues)}")
                model_results["tests"]["multi"] = {
                    "status": status,
                    "turns": [
                        {"tokens": tok1, "time": round(dt1, 1)},
                        {"tokens": tok2, "time": round(dt2, 1)},
                        {"tokens": tok3, "time": round(dt3, 1)},
                    ],
                    "issues": issues
                }

    # ── Test 4: Creative / temp=0.7 long output ──
    print(f"\n  [Test 4] Creative writing (temp=0.7, ~500 tokens)")
    text, tok, dt, err = chat(mid, [
        {"role": "user", "content": "Write a short story about a robot who discovers it can dream. Include dialogue, vivid descriptions, and an unexpected ending. Make it around 300-400 words."}
    ], temp=0.7, max_tokens=700)
    if err:
        print(f"    ERROR: {err}")
        model_results["tests"]["creative"] = {"status": "ERROR", "error": err}
    else:
        issues = check_coherence(text, min_tokens=100)
        status = "FAIL" if issues else "PASS"
        preview = wrap(text[:500])
        print(f"    [{status}] ({tok} tok, {dt:.1f}s)")
        for line in preview.split("\n")[:8]:
            print(f"    | {line}")
        if tok >= 400:
            print(f"    | ... ({tok} tokens total)")
        if issues:
            print(f"    Issues: {', '.join(issues)}")
        model_results["tests"]["creative"] = {"status": status, "tokens": tok, "time": round(dt, 1), "text": text[:500], "issues": issues}

    results[mid] = model_results

# ─── Summary ───
print(f"\n\n{'='*80}")
print(f"  REGRESSION TEST SUMMARY")
print(f"{'='*80}\n")

pass_count = 0
fail_count = 0
error_count = 0
load_fail = 0

for mid, res in results.items():
    if res["status"] == "LOAD_FAILED":
        load_fail += 1
        print(f"  {mid:50s}  LOAD FAILED")
        continue

    tests = res.get("tests", {})
    statuses = []
    for tname, tdata in tests.items():
        s = tdata.get("status", "?")
        statuses.append(s)

    passes = statuses.count("PASS")
    fails = statuses.count("FAIL")
    errors = statuses.count("ERROR")
    pass_count += passes
    fail_count += fails
    error_count += errors

    indicators = []
    for tname in ["short", "long", "multi", "creative"]:
        s = tests.get(tname, {}).get("status", "SKIP")
        if s == "PASS":
            indicators.append("PASS")
        elif s == "FAIL":
            indicators.append("FAIL")
        elif s == "ERROR":
            indicators.append("ERR ")
        else:
            indicators.append("SKIP")

    line = f"  {mid:50s}  " + " | ".join(indicators)
    print(line)

print(f"\n  {'─'*60}")
print(f"  Total: {pass_count} PASS, {fail_count} FAIL, {error_count} ERROR, {load_fail} LOAD_FAILED")
print(f"  Models tested: {len(results)}")
print(f"{'='*80}")

# Write detailed results to JSON
with open("regression_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nDetailed results saved to regression_results.json")

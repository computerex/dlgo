"""
Full coherence test for all models in C:\\models on both GPU and CPU.
Uses a single long-running server with --models-dir, loading/unloading via API.
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PORT = 9091
BASE = f"http://localhost:{PORT}"

# Skip non-LLM models
SKIP_MODELS = {"Kokoro_espeak_F16", "Kokoro_no_espeak_F16", "Nanonets-OCR2-1.5B-exp.i1-Q6_K"}
SKIP_PREFIXES = ["ggml-vocab-", "parakeet-", "whisper-", "small_q8"]

# Models too large for system RAM (skip entirely)
MAX_MODEL_SIZE_MB = 30000


def api(method, path, body=None, timeout=600):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", errors="replace"))
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


def load_model(model_id, path, gpu=True, ctx=4096):
    return api("POST", "/v1/models", {"id": model_id, "path": path, "gpu": gpu, "context": ctx}, timeout=300)


def unload_model(model_id):
    return api("DELETE", "/v1/models", {"id": model_id}, timeout=60)


def chat(model_id, prompt, max_tokens=2000, temp=0.3):
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
        "temperature": temp,
    }
    code, resp = api("POST", "/v1/chat/completions", body, timeout=600)
    if code == 200:
        msg = resp["choices"][0]["message"]["content"]
        usage = resp["usage"]
        finish = resp["choices"][0]["finish_reason"]
        return msg, usage, finish, None
    else:
        err = resp.get("error", {})
        if isinstance(err, dict):
            err = err.get("message", str(err))
        return "", {}, "", str(err)


def get_max_tokens(size_mb):
    if size_mb > 10000:
        return 1000
    if size_mb > 5000:
        return 2000
    return 4000


def check_repetition(text):
    words = text.split()
    for size in [10, 20, 50]:
        for i in range(len(words) - size * 3):
            chunk = " ".join(words[i:i + size])
            count = 0
            j = i
            while j + size <= len(words):
                if " ".join(words[j:j + size]) == chunk:
                    count += 1
                    j += size
                else:
                    break
            if count >= 3:
                return True, f"Repeated {size}-word block {count}x"
    return False, ""


def check_garbage(text):
    if len(text) == 0:
        return True, "Empty output"
    non_printable = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    if non_printable / len(text) > 0.05:
        return True, f"High non-printable ratio"
    return False, ""


def test_numbered_list(model_id, max_tok):
    prompt = (
        "Write a numbered list of exactly 100 interesting science facts. "
        "Number each fact starting from 1. Format: '1. fact text', '2. fact text', etc."
    )
    text, usage, finish, err = chat(model_id, prompt, max_tokens=max_tok, temp=0.3)
    if err:
        return "ERROR", 0, err
    comp = usage.get("completion_tokens", 0)
    g, gm = check_garbage(text)
    if g: return "FAIL", comp, f"Garbage: {gm}"
    r, rm = check_repetition(text)
    if r: return "FAIL", comp, f"Repetition: {rm}"
    numbers = [int(m) for m in re.findall(r'^(\d+)\.', text, re.MULTILINE)]
    if len(numbers) < 10:
        return "FAIL", comp, f"Only {len(numbers)} numbered items"
    errors = sum(1 for i in range(1, len(numbers)) if numbers[i] - numbers[i-1] != 1)
    if errors / max(len(numbers), 1) > 0.15:
        return "FAIL", comp, f"Numbering errors: {errors}/{len(numbers)}"
    return "PASS", comp, f"{len(numbers)} items, {errors} errors"


def test_essay(model_id, max_tok):
    prompt = (
        "Write a detailed essay about the history of the Internet. "
        "Use section headings (## Heading) for at least 4 sections. "
        "Cover: ARPANET, WWW, social media, and future trends."
    )
    text, usage, finish, err = chat(model_id, prompt, max_tokens=max_tok, temp=0.4)
    if err:
        return "ERROR", 0, err
    comp = usage.get("completion_tokens", 0)
    g, gm = check_garbage(text)
    if g: return "FAIL", comp, f"Garbage: {gm}"
    r, rm = check_repetition(text)
    if r: return "FAIL", comp, f"Repetition: {rm}"
    headings = re.findall(r'^#{1,3}\s+(.+)$', text, re.MULTILINE)
    keywords = ["internet", "network", "web", "online", "digital", "data", "computer"]
    found = [k for k in keywords if k in text.lower()]
    if len(found) < 2:
        return "FAIL", comp, f"Off-topic: keywords={found}"
    return "PASS", comp, f"{len(headings)} headings, keywords={found[:5]}"


def test_code(model_id, max_tok):
    prompt = (
        "Write a complete Python implementation of a linked list with methods: "
        "insert, delete, search, reverse, and print_list. "
        "Include docstrings and unit tests."
    )
    text, usage, finish, err = chat(model_id, prompt, max_tokens=max_tok, temp=0.2)
    if err:
        return "ERROR", 0, err
    comp = usage.get("completion_tokens", 0)
    g, gm = check_garbage(text)
    if g: return "FAIL", comp, f"Garbage: {gm}"
    r, rm = check_repetition(text)
    if r: return "FAIL", comp, f"Repetition: {rm}"
    has_class = "class " in text
    has_def = text.count("def ") >= 3
    if not (has_class or has_def):
        return "FAIL", comp, "No class/function definitions"
    return "PASS", comp, f"class={has_class}, defs={text.count('def ')}"


def main():
    # Get available models from server
    print("Fetching available models...")
    code, resp = api("GET", "/v1/models")
    if code != 200:
        print(f"ERROR: Cannot reach server: {resp}")
        sys.exit(1)

    available = {m["id"]: m["path"] for m in resp.get("available", [])}
    loaded = {m["id"]: m.get("path", "") for m in resp.get("data", [])}
    all_models = {**available, **loaded}

    # Filter and sort by file size
    models = []
    for mid, mpath in all_models.items():
        if mid in SKIP_MODELS or any(mid.startswith(p) for p in SKIP_PREFIXES):
            continue
        try:
            size_mb = os.path.getsize(mpath) / (1024 * 1024) if os.path.exists(mpath) else 0
        except Exception:
            size_mb = 0
        if size_mb > MAX_MODEL_SIZE_MB:
            print(f"  SKIP {mid} ({size_mb:.0f} MB > {MAX_MODEL_SIZE_MB} MB)")
            continue
        models.append({"id": mid, "path": mpath, "size_mb": size_mb})

    models.sort(key=lambda m: m["size_mb"])
    print(f"Testing {len(models)} models (smallest first):")
    for m in models:
        print(f"  {m['id']} ({m['size_mb']:.0f} MB)")

    all_results = {}

    for i, model in enumerate(models):
        mid = model["id"]
        mpath = model["path"]
        size = model["size_mb"]
        max_tok = get_max_tokens(size)

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(models)}] {mid} ({size:.0f} MB, max_tokens={max_tok})")
        print(f"{'='*70}")
        sys.stdout.flush()

        model_results = {}

        # Determine backends: GPU if fits, always CPU
        backends = []
        if size < 15000:  # ~15GB VRAM limit
            backends.append(("GPU", True))
        else:
            print(f"  Skipping GPU (too large for VRAM)")
        backends.append(("CPU", False))

        for backend_name, use_gpu in backends:
            print(f"\n  --- {backend_name} ---")

            # Unload any currently loaded models
            _, cur = api("GET", "/v1/models")
            for lm in cur.get("data", []):
                print(f"  Unloading previously loaded: {lm['id']}")
                unload_model(lm["id"])
                time.sleep(2)

            # Load model
            print(f"  Loading {mid} ({backend_name})...", end=" ", flush=True)
            t0 = time.time()
            lcode, lresp = load_model(mid, mpath, gpu=use_gpu, ctx=4096)
            elapsed = time.time() - t0

            if lcode != 200:
                err = lresp.get("error", {})
                if isinstance(err, dict):
                    err = err.get("message", str(err))
                print(f"FAILED ({lcode}): {err}")
                model_results[backend_name] = {"load_error": str(err)[:200]}
                continue
            print(f"OK ({elapsed:.1f}s)")

            # Run tests
            test_results = {}
            for test_name, test_fn in [("numbered_list", test_numbered_list),
                                        ("essay", test_essay),
                                        ("code", test_code)]:
                t0 = time.time()
                status, tokens, detail = test_fn(mid, max_tok)
                dt = time.time() - t0
                test_results[test_name] = {"status": status, "tokens": tokens, "detail": detail, "time_s": round(dt, 1)}
                marker = {"PASS": "PASS", "FAIL": "FAIL"}.get(status, "ERR ")
                print(f"    [{marker}] {test_name}: {tokens} tokens ({dt:.1f}s) - {detail}")
                sys.stdout.flush()

            model_results[backend_name] = test_results

            # Unload
            print(f"  Unloading {mid}...", end=" ", flush=True)
            unload_model(mid)
            time.sleep(2)
            print("OK")

        all_results[mid] = model_results

        # Save incrementally
        with open("coherence_all_models.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n\n{'='*70}")
    print("COHERENCE TEST SUMMARY")
    print(f"{'='*70}")
    for mname, backends in all_results.items():
        for bname, tests in backends.items():
            if "load_error" in tests:
                print(f"  {mname:50s} {bname:4s} LOAD_FAILED: {tests['load_error'][:60]}")
            else:
                passes = sum(1 for t in tests.values() if t.get("status") == "PASS")
                total = len(tests)
                tot_tok = sum(t.get("tokens", 0) for t in tests.values())
                print(f"  {mname:50s} {bname:4s} {passes}/{total} PASS ({tot_tok} tokens)")

    print(f"\nResults: coherence_all_models.json")


if __name__ == "__main__":
    main()

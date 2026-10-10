"""32k-token generation soak test for the 9B GPU server.

Sends one non-thinking, temperature=0 request with max_tokens=32768 and saves
the full completion. Verifies finish_reason/GeneratedTokens and dumps
sanity metadata (length, repetition stats) for coherence review.

Usage: python bench/systemone/test_32k_generation.py [port] [out_file]
"""
import json
import re
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8097
OUT = sys.argv[2] if len(sys.argv) > 2 else "bench/systemone/gen32k-output.txt"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "qwen35-9b"
MAX_TOKENS = 32768

PROMPTS = {
    "essay": (
        "Write a comprehensive, well-structured technical essay titled "
        "'The Complete History of Computing: From Babbage to Modern AI'. "
        "Cover each era in detail with dated milestones, key figures, specific "
        "machines, and their impact. Use markdown headings per era. Do not repeat "
        "yourself; keep advancing the narrative. Continue until the history reaches "
        "modern large language models."
    ),
    # ~80k tokens of work > 32768 budget -> finish_reason "length", guaranteed
    # full-length generation with a machine-checkable sequence.
    "count": (
        "List every integer from 1 to 50000 in ascending order, separated by "
        "commas. Do not skip, summarize, or abbreviate. Output only the numbers."
    ),
}


def main():
    which = sys.argv[3] if len(sys.argv) > 3 else "essay"
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[which]}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as resp:
        payload = json.load(resp)
    ms = (time.perf_counter() - t0) * 1000

    choice = payload["choices"][0]
    content = choice["message"].get("content", "") or ""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(content)

    # --- sanity metrics ---
    words = content.split()
    # longest repeated substring proxy: most common 8-word shingle
    shingles = {}
    ws = [w.lower() for w in words]
    for i in range(len(ws) - 8):
        key = " ".join(ws[i : i + 8])
        shingles[key] = shingles.get(key, 0) + 1
    top = sorted(shingles.items(), key=lambda kv: -kv[1])[:3]
    weird = sum(1 for ch in content if ord(ch) > 0x3000)

    # Sequence integrity for the counting task: extract integers, verify
    # strictly monotonic +1 chain from 1.
    nums = [int(m) for m in re.findall(r"\d+", content)]
    seq_ok, seq_break = True, None
    if which == "count" and nums:
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] + 1:
                seq_ok, seq_break = False, (i, nums[i - 1], nums[i])
                break
        if nums[0] != 1:
            seq_ok, seq_break = False, (0, "start", nums[0])

    print(json.dumps({
        "task": which,
        "finish_reason": choice.get("finish_reason"),
        "generated_tokens": payload.get("usage", {}).get("completion_tokens"),
        "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"),
        "chars": len(content),
        "words": len(words),
        "elapsed_s": round(ms / 1000, 1),
        "tok_per_s": round((payload.get("usage", {}).get("completion_tokens") or 0) / (ms / 1000), 1),
        "max_8gram_repeat": top[0] if top else None,
        "cjk_chars": weird,
        "headings": len(re.findall(r"^#{1,3} ", content, re.M)),
        "seq_first": nums[0] if nums else None,
        "seq_last": nums[-1] if nums else None,
        "seq_monotonic": seq_ok if which == "count" else None,
        "seq_break": seq_break,
    }, indent=2))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()

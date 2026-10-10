"""Ask the chess battery questions through normal chat completion (not System One)."""
import json
import time
import urllib.request
import sys

URL = "http://127.0.0.1:8097/v1/chat/completions"
REQUEST = "systemone-train/request-chess-multi.json"
MODEL = "qwen35-9b"


def build_prompt(body):
    parts = []
    for qid, q in body["questions"].items():
        if q["type"] == "choice":
            opts = "; ".join("%s: %s" % (k, v) for k, v in q["criteria"].items())
            parts.append("%s Options: %s" % (q["instructions"], opts))
        elif q["type"] == "noul":
            parts.append(q["instructions"] + " Answer yes or no.")
        else:
            opts = "; ".join("%d: %s" % (i, c) for i, c in enumerate(q["criteria"]))
            parts.append("%s Options: %s" % (q["instructions"], opts))
    numbered = "\n".join("%d. %s" % (i + 1, t) for i, t in enumerate(parts))
    return "State: " + body["state"] + "\n\n" + numbered


def main():
    thinking = "--thinking" in sys.argv
    max_tokens = 4096 if thinking else 512
    body = json.load(open(REQUEST, encoding="utf-8"))
    prompt = build_prompt(body)
    req_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(URL, json.dumps(req_body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    latency = (time.perf_counter() - t0) * 1000
    content = payload["choices"][0]["message"]["content"]
    print("thinking=%s latency_ms=%.0f finish=%s" %
          (thinking, latency, payload["choices"][0].get("finish_reason")))
    print("--- response ---")
    print(content[:2000])


if __name__ == "__main__":
    main()

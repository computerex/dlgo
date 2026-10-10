import json
import time
import urllib.request

URL = "http://127.0.0.1:8097/v1/systemone"
READOUT = r"C:\projects\dlgo\systemone-train\artifacts\qwen35-9b-q3-readout-v1-gpu.json"

body = {
    "model": "qwen35-9b",
    "readout": READOUT,
    "state": "Benchmark nonce 9bdiag01. Your team owns a chat service. Users report the mobile app crashes when opening the attachments panel on Android 12. The crash rate tripled after last week's release. The attachments panel was rewritten in the release.",
    "questions": {
        "category": {
            "type": "choice",
            "instructions": "Which category does this issue belong to?",
            "criteria": {
                "bug": "A defect in shipped software behavior.",
                "feature_request": "A request for new functionality.",
                "billing": "A payment or invoice issue.",
                "how_to": "A usage question.",
                "security": "A vulnerability or abuse report.",
            },
        },
        "needs_urgent_reply": {
            "type": "noul",
            "instructions": "Does this report need an urgent reply?",
        },
    },
}

req = urllib.request.Request(URL, json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as resp:
    payload = json.load(resp)
print(f"elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}")
print("answers:", list(payload.get("answers", {}).keys()))

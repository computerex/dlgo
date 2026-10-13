#!/usr/bin/env python3
"""Compare Jev 1.13 and local System One on an identical Pokemon noul request.

Reads OPENROUTER_API_KEY from systemone-train/.env or environment. It never
prints the key. Results are written to an ignored JSON artifact by default.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "systemone-train" / ".env"
OUT = ROOT / "bench" / "results" / "jev-pokemon-comparison.json"
LOCAL = "http://127.0.0.1:8080/v1/systemone"
READOUT = str(ROOT / "systemone-train" / "artifacts" / "qwen35-9b-q3-readout-v2-uni.json")

STATE = ("You are in a Pokemon battle. The opponent's active Pokemon is Blastoise, a Water-type turtle that uses "
         "powerful Water attacks. Your active Pokemon is Charizard, a Fire/Flying-type at half health. Your party is: "
         "Charizard (Fire/Flying), Pikachu (Electric), Gengar (Ghost/Poison), and Snorlax (Normal).")
QUESTION = "Is your active Charizard, a Fire/Flying-type, weak to the opponent's Water attacks?"

def api_key():
    # Prefer the project-local ignored secret over an inherited shell value;
    # the latter may be a stale key from another project/session.
    key = ""
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
                break
    key = key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing (set it or add it to systemone-train/.env)")
    return key

def post(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read()), round((time.perf_counter() - started) * 1000, 1), dict(response.headers)

def main():
    question = {"type": "noul", "instructions": QUESTION,
                "criteria": {"true": "Water attacks deal super-effective damage to Charizard.",
                             "false": "Water attacks deal neutral, reduced, or no damage to Charizard."}}
    jev, jev_ms, _ = post("https://openrouter.ai/api/alpha/decisions", {
        "model": "typesafe/jev-1.13", "state": STATE,
        "questions": {"charizard_water_weak": question},
    }, {"Authorization": "Bearer " + api_key(), "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", "X-Title": "dlgo Jev parity diagnostic"})
    local, local_ms, _ = post(LOCAL, {
        "model": "qwen35-9b", "readout": READOUT, "state": STATE,
        "questions": {"charizard_water_weak": {"type": "noul", "instructions": QUESTION}},
    }, {"Content-Type": "application/json"}, timeout=300)
    comparison = {
        "case": {"truth": "yes", "state": STATE, "instructions": QUESTION},
        "jev": {"model": jev.get("model"), "answer": jev["answers"]["charizard_water_weak"],
                "usage": jev.get("usage"), "wall_ms": jev_ms},
        "dlgo_9b_v2": {"readout": READOUT, "answer": local["answers"]["charizard_water_weak"],
                       "usage": local.get("usage"), "wall_ms": local_ms},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print("Jev p(yes):", comparison["jev"]["answer"].get("noul"), "wall_ms:", jev_ms)
    print("dlgo v2 p(yes):", comparison["dlgo_9b_v2"]["answer"].get("noul"), "wall_ms:", local_ms)
    print("saved:", OUT)

if __name__ == "__main__":
    main()

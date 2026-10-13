#!/usr/bin/env python3
"""Probe: same knowledge, choice-format (in-distribution) vs noul-format (out-of-distribution)."""
import json, urllib.request, sys

BASE = "http://127.0.0.1:8080"
STATE = ("You are in a Pokemon battle. The opponent's active Pokemon is Blastoise, a Water-type turtle that uses "
         "powerful Water attacks. Your active Pokemon is Charizard, a Fire/Flying-type at half health. Your party is: "
         "Charizard (Fire/Flying), Pikachu (Electric), Gengar (Ghost/Poison), and Snorlax (Normal).")

def decide(model, readout, questions):
    body = json.dumps({"model": model, "readout": readout, "state": STATE, "questions": questions}).encode()
    req = urllib.request.Request(BASE + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

model = sys.argv[1] if len(sys.argv) > 1 else "qwen35-9b"
readout = {
    "qwen35-9b":  "C:\\projects\\dlgo\\systemone-train\\artifacts\\qwen35-9b-q3-readout-v2-uni.json",
    "qwen38-2b":  "C:\\projects\\dlgo\\systemone-train\\artifacts\\qwen38-2b-q4-readout-v2-uni.json",
    "qwen35-08b": "C:\\projects\\dlgo\\systemone-train\\artifacts\\qwen35-08b-q4-readout-v10-uni.json",
}[model]

qs = {
    # noul format (bare yes/no) — the playground shape
    "noul_weak": {"type": "noul", "instructions": "Is your active Charizard, a Fire/Flying-type, weak to the opponent's Water attacks?"},
    # same question, choice format WITH descriptions — the training shape
    "choice_weak": {
        "type": "choice",
        "instructions": "Is your active Charizard, a Fire/Flying-type, weak to the opponent's Water attacks?",
        "criteria": {
            "yes": "Water attacks deal super-effective (double) damage to Charizard.",
            "no": "Water attacks deal normal or reduced damage to Charizard.",
        },
    },
    # choice format WITHOUT descriptions (labels default to the raw values)
    "choice_bare_weak": {
        "type": "choice",
        "instructions": "Is your active Charizard, a Fire/Flying-type, weak to the opponent's Water attacks?",
        "criteria": {"no": "no", "yes": "yes"},
    },
    # more knowledge checks in choice-with-description format
    "blastoise_electric": {
        "type": "choice",
        "instructions": "Is the opponent's Blastoise, a Water-type, weak to Electric attacks?",
        "criteria": {"yes": "Electric attacks deal super-effective (double) damage to Water-type Pokemon.", "no": "Electric attacks deal normal or reduced damage to Water-type Pokemon."},
    },
    "gengar_psychic": {
        "type": "choice",
        "instructions": "Is your Gengar, a Ghost/Poison-type, weak to Psychic attacks?",
        "criteria": {"yes": "Psychic attacks deal super-effective (double) damage to Ghost/Poison-type Pokemon.", "no": "Psychic attacks deal normal or reduced damage to Ghost/Poison-type Pokemon."},
    },
    "pikachu_ground": {
        "type": "choice",
        "instructions": "Is your Pikachu, an Electric-type, weak to Ground attacks?",
        "criteria": {"yes": "Ground attacks deal super-effective (double) damage to Electric-type Pokemon.", "no": "Ground attacks deal no damage to Electric-type Pokemon (they are immune)."},
    },
}
out = decide(model, readout, qs)
for qid in sorted(out["answers"]):
    a = out["answers"][qid]
    if a["type"] == "noul":
        print(f"{qid:22s} noul   p(yes)={a['noul']:.4f}")
    else:
        probs = a["probabilities"]
        print(f"{qid:22s} choice {a['choice']:4s} p={probs[a['choice']]:.4f}  ({' '.join(f'{k}={v:.3f}' for k, v in sorted(probs.items()))})")

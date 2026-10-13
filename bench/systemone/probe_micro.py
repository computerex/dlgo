#!/usr/bin/env python3
"""Micro-tests: (1) bare-choice alone vs noul alone (multi-path interference check),
(2) noul with criteria descriptions, (3) backbone generation knowledge."""
import json, urllib.request, sys

BASE = "http://127.0.0.1:8080"
STATE = ("You are in a Pokemon battle. The opponent's active Pokemon is Blastoise, a Water-type turtle that uses "
         "powerful Water attacks. Your active Pokemon is Charizard, a Fire/Flying-type at half health. Your party is: "
         "Charizard (Fire/Flying), Pikachu (Electric), Gengar (Ghost/Poison), and Snorlax (Normal).")

def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

model = "qwen35-9b"
readout = "C:\\projects\\dlgo\\systemone-train\\artifacts\\qwen35-9b-q3-readout-v2-uni.json"
INSTR = "Is your active Charizard, a Fire/Flying-type, weak to the opponent's Water attacks?"

def run(tag, questions):
    out = post("/v1/systemone", {"model": model, "readout": readout, "state": STATE, "questions": questions})
    for qid in sorted(out["answers"]):
        a = out["answers"][qid]
        if a["type"] == "noul":
            print(f"{tag:34s} noul   p(yes)={a['noul']:.4f}")
        else:
            p = a["probabilities"]
            print(f"{tag:34s} choice {a['choice']:4s} p={p[a['choice']]:.4f}")

# 1. bare choice ALONE (noul compiles to the identical prompt)
run("bare-choice ALONE", {"choice_bare_weak": {"type": "choice", "instructions": INSTR, "criteria": {"no": "no", "yes": "yes"}}})
# 2. noul ALONE (reference)
run("noul ALONE", {"noul_weak": {"type": "noul", "instructions": INSTR}})
# 3. noul WITH criteria descriptions
run("noul WITH descriptions", {"noul_desc": {"type": "noul", "instructions": INSTR, "criteria": {
    "true": "Water attacks deal super-effective (double) damage to Charizard.",
    "false": "Water attacks deal normal or reduced damage to Charizard.",
}}})
# 4. bare choice + noul TOGETHER in one batch (interference check vs ALONE numbers)
run("bare+noul SAME BATCH", {
    "choice_bare_weak": {"type": "choice", "instructions": INSTR, "criteria": {"no": "no", "yes": "yes"}},
    "noul_weak": {"type": "noul", "instructions": INSTR},
})

# 5. backbone generation knowledge
chat = {
    "model": model,
    "messages": [{"role": "user", "content": "Answer with only yes or no. In Pokemon, is Charizard (a Fire/Flying-type Pokemon) weak to Water-type attacks?"}],
    "temperature": 0, "max_tokens": 8, "stream": False,
}
gen = post("/v1/chat/completions", chat, timeout=300)
print("backbone generation:", repr(gen["choices"][0]["message"]["content"].strip()))
chat["messages"][0]["content"] = "Answer with only yes or no. In Pokemon, is Blastoise (a Water-type Pokemon) weak to Electric-type attacks?"
gen = post("/v1/chat/completions", chat, timeout=300)
print("backbone generation:", repr(gen["choices"][0]["message"]["content"].strip()))

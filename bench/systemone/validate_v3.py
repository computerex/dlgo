#!/usr/bin/env python3
"""Post-fix validation suite for a 9B readout artifact.
Usage: python validate_v3.py <model-id> <readout.json>
Checks: charizard noul (truth yes), choice-format, held-out matchups
(blastoise/electric truth yes, pikachu/ground truth no-immune), the full
playground preset (pikachu switch), and battery-style knowledge nouls.
"""
import json, urllib.request, sys

BASE = "http://127.0.0.1:8080"
POKEMON_STATE = ("You are in a Pokemon battle. The opponent's active Pokemon is Blastoise, a Water-type turtle that uses "
                 "powerful Water attacks. Your active Pokemon is Charizard, a Fire/Flying-type at half health. Your party is: "
                 "Charizard (Fire/Flying), Pikachu (Electric), Gengar (Ghost/Poison), and Snorlax (Normal).")

def decide(state, questions, timeout=300):
    body = json.dumps({"model": MODEL, "readout": READOUT, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(BASE + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen35-9b"
READOUT = sys.argv[2] if len(sys.argv) > 2 else "C:\\projects\\dlgo\\systemone-train\\artifacts\\qwen35-9b-q3-readout-v3-uni.json"

failures = 0
def check(name, ok, detail):
    global failures
    mark = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{mark}] {name}: {detail}")

# --- the originally-failing question ---
out = decide(POKEMON_STATE, {"charizard_water_weak": {"type": "noul",
    "instructions": "Is your active Charizard, a Fire/Flying-type, weak to the opponent's Water attacks?"}})
p = out["answers"]["charizard_water_weak"]["noul"]
check("charizard_water_weak (noul, truth yes)", p >= 0.5, f"p(yes)={p:.4f}")

# --- held-out type-chart matchups (never in training) ---
heldout = [
    ("blastoise_electric", "Is the opponent's Blastoise, a Water-type, weak to Electric-type attacks?", True),
    ("pikachu_ground", "Is your Pikachu, an Electric-type, weak to Ground attacks?", False),
    ("gengar_psychic", "Is your Gengar, a Ghost/Poison-type, weak to Psychic attacks?", True),
    ("snorlax_psychic", "Is your Snorlax, a Normal-type, weak to Psychic attacks?", False),
    ("gyarados_electric", "Is a Gyarados, a Water/Flying-type, weak to Electric attacks?", True),
    ("onix_electric", "Is an Onix, a Rock/Ground-type, weak to Electric attacks?", False),
]
qs = {qid: {"type": "noul", "instructions": instr} for qid, instr, _ in heldout}
out = decide(POKEMON_STATE, qs)
for qid, _, truth in heldout:
    p = out["answers"][qid]["noul"]
    ok = p >= 0.5 if truth else p < 0.5
    check(f"{qid} (noul, truth {'yes' if truth else 'no'})", ok, f"p(yes)={p:.4f}")

# --- choice format must stay healthy ---
out = decide(POKEMON_STATE, {"choice_weak": {"type": "choice",
    "instructions": "Is your active Charizard, a Fire/Flying-type, weak to the opponent's Water attacks?",
    "criteria": {"no": "Water attacks deal normal or reduced damage to Charizard.",
                 "yes": "Water attacks deal super-effective (double) damage to Charizard."}}})
a = out["answers"]["choice_weak"]
check("charizard choice-format", a["choice"] == "yes", f"choice={a['choice']} p={a['probabilities'][a['choice']]:.4f}")

# --- full playground preset (switch must still be pikachu) ---
out = decide(POKEMON_STATE, {
    "switch": {"type": "choice", "instructions": "Which party member should you switch in to gain the best type advantage against the opponent's Water-type Blastoise?",
               "criteria": {"pikachu": "Electric-type Pokemon; Electric attacks deal super-effective damage to Water Pokemon.",
                            "gengar": "Ghost/Poison-type Pokemon; its attacks have no type advantage against Water.",
                            "snorlax": "Normal-type Pokemon; its attacks have no type advantage against Water."}},
    "move_type": {"type": "choice", "instructions": "Which move type deals the most damage to the opponent's Water-type Blastoise?",
                  "criteria": {"electric": "Electric moves deal super-effective (double) damage to Water Pokemon.",
                               "fire": "Fire moves deal resisted (half) damage to Water Pokemon.",
                               "water": "Water moves deal neutral damage to Water Pokemon."}},
})
sw = out["answers"]["switch"]; mt = out["answers"]["move_type"]
check("preset switch", sw["choice"] == "pikachu", f"choice={sw['choice']} p={sw['probabilities'][sw['choice']]:.4f}")
check("preset move_type", mt["choice"] == "electric", f"choice={mt['choice']} p={mt['probabilities'][mt['choice']]:.4f}")

print(f"\n{'ALL CHECKS PASS' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)

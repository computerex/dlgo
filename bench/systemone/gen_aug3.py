#!/usr/bin/env python3
"""Add focused, held-out-safe Pokemon noul generalization examples to train-aug2.

This deliberately never adds the validation pairs (Charizard/Water or
Blastoise/Electric).  It teaches the same type relationships through other
species, particularly Fire/Flying + Water and neutral/immune negatives.
"""
import json

BASE = "systemone-train/data/generated/splits-gp2"
IN = f"{BASE}/train-aug2.jsonl"
OUT = f"{BASE}/train-aug3.jsonl"

# species, displayed types, attack, truth, explanation.  The set is balanced
# and covers dual-type multipliers, immunities, neutral interactions, and
# resistances.  None is a validation question or its answer pair.
MATCHUPS = [
    ("Moltres", "Fire/Flying", "Water", "yes", "Water is super-effective against Fire and neutral against Flying."),
    ("Talonflame", "Fire/Flying", "Water", "yes", "Water is super-effective against Fire and neutral against Flying."),
    ("Ho-Oh", "Fire/Flying", "Water", "yes", "Water is super-effective against Fire and neutral against Flying."),
    ("Moltres", "Fire/Flying", "Electric", "yes", "Electric is super-effective against Flying."),
    ("Talonflame", "Fire/Flying", "Rock", "yes", "Rock is super-effective against both Fire and Flying."),
    ("Skarmory", "Steel/Flying", "Ground", "no", "Flying Pokemon are immune to Ground attacks."),
    ("Pidgeot", "Normal/Flying", "Ground", "no", "Flying Pokemon are immune to Ground attacks."),
    ("Gyarados", "Water/Flying", "Ground", "no", "Flying Pokemon are immune to Ground attacks."),
    ("Raichu", "Electric", "Ground", "no", "Ground attacks are super-effective against Electric Pokemon."),
    ("Jolteon", "Electric", "Ground", "no", "Ground attacks are super-effective against Electric Pokemon."),
    ("Magnemite", "Electric/Steel", "Ground", "yes", "Ground is super-effective against Electric and Steel."),
    ("Eevee", "Normal", "Psychic", "no", "Psychic attacks deal neutral damage to Normal Pokemon."),
    ("Tauros", "Normal", "Psychic", "no", "Psychic attacks deal neutral damage to Normal Pokemon."),
    ("Raticate", "Normal", "Psychic", "no", "Psychic attacks deal neutral damage to Normal Pokemon."),
    ("Machoke", "Fighting", "Psychic", "yes", "Psychic attacks are super-effective against Fighting Pokemon."),
    ("Hitmonlee", "Fighting", "Psychic", "yes", "Psychic attacks are super-effective against Fighting Pokemon."),
    ("Vaporeon", "Water", "Electric", "yes", "Electric attacks are super-effective against Water Pokemon."),
    ("Starmie", "Water/Psychic", "Electric", "yes", "Electric attacks are super-effective against Water Pokemon."),
    ("Lapras", "Water/Ice", "Electric", "yes", "Electric attacks are super-effective against Water Pokemon."),
    ("Ninetales", "Fire", "Water", "yes", "Water attacks are super-effective against Fire Pokemon."),
    ("Arcanine", "Fire", "Water", "yes", "Water attacks are super-effective against Fire Pokemon."),
    ("Vulpix", "Fire", "Grass", "no", "Fire attacks resist Grass attacks."),
    ("Sandslash", "Ground", "Electric", "no", "Ground Pokemon are immune to Electric attacks."),
    ("Marowak", "Ground", "Electric", "no", "Ground Pokemon are immune to Electric attacks."),
]

TEMPLATES = [
    "Is {name}, a {types}-type Pokemon, weak to {attack}-type attacks?",
    "Would a {attack}-type attack be super-effective against {name} ({types})?",
    "In a standard Pokemon battle, is {name} ({types}) vulnerable to {attack} attacks?",
    "Does {name}, whose type is {types}, take super-effective damage from {attack}?",
    "Is it correct that {attack}-type moves are a weakness of {name}, a {types}-type?",
    "Against a {types}-type {name}, are {attack}-type attacks super-effective?",
]

with open(IN, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
seen = {r["id"] for r in rows}
new = []
for species, types, attack, target, fact in MATCHUPS:
    for n, template in enumerate(TEMPLATES):
        ident = f"aug3-typechart-{species}-{attack}-{n}".lower()
        if ident in seen:
            raise SystemExit(f"duplicate id: {ident}")
        state = ("You are deciding a Pokemon type matchup under the standard Pokemon type chart. "
                 f"The defending Pokemon is {species}, which is {types}-type. "
                 "Use its complete listed type combination when determining effectiveness.")
        instruction = template.format(name=species, types=types, attack=attack)
        # Bare labels exactly match the production noul compiler.
        new.append({"id": ident, "task": "choice", "family": "pokemon_typechart_aug3",
                    "split_role": "train", "source": "aug3-heldout-safe-typechart",
                    "state": state, "instructions": instruction,
                    "criteria": {"no": "no", "yes": "yes"},
                    "codes": {"no": "A", "yes": "B"}, "target": target})
        described = ident + "-described"
        no = "The attack is resisted, neutral, or has no effect; it is not super-effective."
        yes = "The attack deals super-effective damage."
        new.append({"id": described, "task": "choice", "family": "pokemon_typechart_aug3",
                    "split_role": "train", "source": "aug3-heldout-safe-typechart",
                    "state": state + " Type-chart fact: " + fact,
                    "instructions": instruction,
                    "criteria": {"no": no, "yes": yes},
                    "codes": {"no": "A", "yes": "B"}, "target": target})

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for row in rows + new:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
labels = {x: sum(r["target"] == x for r in new) for x in ("yes", "no")}
print(f"base={len(rows)} new={len(new)} labels={labels} total={len(rows)+len(new)} -> {OUT}")

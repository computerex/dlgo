#!/usr/bin/env python3
"""Generate augmented System One training rows (v2 corpus -> train-aug2.jsonl).

Adds what the v2 corpus lacks entirely:
  1. noul-format rows (bare yes/no) derived from existing choice rows — teaches
     the bare `A: no / B: yes` format the server emits for criteria-less noul.
  2. described verification rows — choice-style labels for the same skill.
  3. Pokemon type-chart knowledge rows — the exact failure class observed
     (world-knowledge verification). The two observed wrong questions
     (charizard/water, blastoise/electric) are deliberately EXCLUDED so
     post-fix correctness proves generalization.
  4. score-format rows derived from ordinal families (priority, severity, ...).
  5. hand-verified battery rows (chess, Yu-Gi-Oh, GDPR) in all three formats.

Row schema matches the corpus exactly (task/family/state/instructions/
criteria/codes/target); criteria and codes are JSON strings. All new ids are
prefixed "aug2-" and derived ONLY from train-split states (no test leakage).
"""
import json, random, collections, sys

random.seed(20261010)
BASE = "systemone-train/data/generated/splits-gp2"
OUT = f"{BASE}/train-aug2.jsonl"

rows = []
def add(id_, family, state, instructions, criteria, target, codes=None, source="aug2-synthetic"):
    # The GPU feature extractor requires explicit answer codes; assign stable
    # code letters by sorted value (matches the compiler's auto-assignment).
    if codes is None:
        codes = {v: chr(ord('A') + i) for i, v in enumerate(sorted(criteria.keys()))}
    rows.append({
        "id": f"aug2-{id_}",
        "task": "choice",
        "family": family,
        "split_role": "train",
        "source": source,
        "state": state,
        "instructions": instructions,
        "criteria": criteria,
        "codes": codes,
        "target": target,
    })

# ---------------------------------------------------------------------------
# Load base corpus
base_rows = []
with open(f"{BASE}/train-aug.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            crit = r["criteria"]
            r["_crit"] = json.loads(crit) if isinstance(crit, str) else crit
            base_rows.append(r)
print(f"base rows: {len(base_rows)}")

# ---------------------------------------------------------------------------
# 1+2. Derived verification (noul) rows from choice rows.
# Balance: one yes-row (proposed = target) + one no-row (proposed = wrong option).
VERIFY_SAMPLES = 800
sampled = random.sample(base_rows, min(VERIFY_SAMPLES, len(base_rows)))
n_yes = n_no = n_desc = 0
for i, r in enumerate(sampled):
    instr = r["instructions"].strip()
    crit = r["_crit"]
    keys = sorted(crit.keys())
    target = r["target"]
    wrongs = [k for k in keys if k != target]
    if not wrongs or not instr:
        continue
    wrong = random.choice(wrongs)

    proposals = [(target, "yes"), (wrong, "no")]
    random.shuffle(proposals)
    for value, label in proposals:
        phrasing = random.choice([
            f"Question: {instr}\nProposed answer: {value}. Is the proposed answer correct?",
            f"Based on the state, evaluate this proposal. {instr} Proposed answer: \"{value}\". Is it correct?",
            f"A decision system reviewed the state and answered \"{value}\" to the following: {instr} Did it answer correctly?",
        ])
        add(f"verify-{i}-{label}", f"{r['family']}-verify", r["state"], phrasing,
            {"no": "no", "yes": "yes"}, label)
        if label == "yes":
            n_yes += 1
        else:
            n_no += 1

    # described-labels variant for a subset (teaches described verification too)
    if i % 4 == 0:
        value, label = proposals[0]
        phrasing = f"Question: {instr}\nProposed answer: {value} ({crit[value]}). Is the proposed answer correct?"
        add(f"verifydesc-{i}", f"{r['family']}-verify", r["state"], phrasing,
            {"no": "The proposed answer is not correct.", "yes": "The proposed answer is correct."}, label)
        n_desc += 1
print(f"derived noul: yes={n_yes} no={n_no} described={n_desc}")

# ---------------------------------------------------------------------------
# 3. Pokemon type-chart knowledge rows.
# (name, types, attack, verdict) — verdict: yes (weak), no (resist/immune/neutral).
# Effectiveness per the standard type chart; immune/resist cases verified by hand.
MATCHUPS = [
    ("Pikachu", "Electric", "Ground", "no", "immune"),
    ("Pikachu", "Electric", "Water", "yes", "se"),
    ("Pikachu", "Electric", "Fighting", "no", "neutral"),
    ("Gengar", "Ghost/Poison", "Normal", "no", "immune"),
    ("Gengar", "Ghost/Poison", "Fighting", "no", "immune"),
    ("Gengar", "Ghost/Poison", "Psychic", "yes", "se"),
    ("Gengar", "Ghost/Poison", "Ground", "no", "neutral"),
    ("Gengar", "Ghost/Poison", "Dark", "yes", "se"),
    ("Gengar", "Ghost/Poison", "Poison", "no", "resist"),
    ("Onix", "Rock/Ground", "Water", "yes", "se"),
    ("Onix", "Rock/Ground", "Electric", "no", "immune"),
    ("Onix", "Rock/Ground", "Grass", "yes", "se"),
    ("Onix", "Rock/Ground", "Poison", "no", "neutral"),
    ("Snorlax", "Normal", "Fighting", "yes", "se"),
    ("Snorlax", "Normal", "Psychic", "no", "neutral"),
    ("Snorlax", "Normal", "Rock", "no", "neutral"),
    ("Venusaur", "Grass/Poison", "Fire", "yes", "se"),
    ("Venusaur", "Grass/Poison", "Psychic", "yes", "se"),
    ("Venusaur", "Grass/Poison", "Ground", "no", "neutral"),
    ("Charizard", "Fire/Flying", "Rock", "yes", "se"),
    ("Charizard", "Fire/Flying", "Electric", "yes", "se"),
    ("Charizard", "Fire/Flying", "Fire", "no", "resist"),
    ("Charizard", "Fire/Flying", "Grass", "no", "resist"),
    ("Blastoise", "Water", "Grass", "yes", "se"),
    ("Blastoise", "Water", "Water", "no", "resist"),
    ("Blastoise", "Water", "Normal", "no", "neutral"),
    ("Gyarados", "Water/Flying", "Electric", "yes", "se"),
    ("Gyarados", "Water/Flying", "Rock", "yes", "se"),
    ("Dragonite", "Dragon/Flying", "Ice", "yes", "se"),
    ("Alakazam", "Psychic", "Ghost", "yes", "se"),
    ("Alakazam", "Psychic", "Dark", "no", "immune"),
    ("Alakazam", "Psychic", "Psychic", "no", "resist"),
    ("Machamp", "Fighting", "Psychic", "yes", "se"),
    ("Machamp", "Fighting", "Flying", "yes", "se"),
    ("Machamp", "Fighting", "Bug", "no", "resist"),
    ("Nidoqueen", "Poison/Ground", "Water", "yes", "se"),
    ("Nidoqueen", "Poison/Ground", "Ice", "yes", "se"),
    ("Geodude", "Rock/Ground", "Grass", "yes", "se"),
    ("Lapras", "Water/Ice", "Electric", "yes", "se"),
    ("Butterfree", "Bug/Flying", "Rock", "yes", "se"),
    ("Scyther", "Bug/Flying", "Fire", "no", "neutral"),
    ("Aerodactyl", "Rock/Flying", "Electric", "yes", "se"),
    ("Arbok", "Poison", "Ground", "yes", "se"),
    ("Magikarp", "Water", "Fire", "no", "resist"),
    ("Golduck", "Water", "Psychic", "no", "neutral"),
    ("Golem", "Rock/Ground", "Electric", "no", "immune"),
]
POKEMON_STATE = ("You are in a Pokemon battle. The opponent's active Pokemon is Blastoise, a Water-type turtle that uses "
                 "powerful Water attacks. Your active Pokemon is Charizard, a Fire/Flying-type at half health. Your party is: "
                 "Charizard (Fire/Flying), Pikachu (Electric), Gengar (Ghost/Poison), and Snorlax (Normal).")
YES_SE = "{attack} attacks deal super-effective (double or more) damage to {name}."
NO_IMMUNE = "{attack} attacks have no effect on {name}."
NO_RESIST = "{attack} attacks are not very effective against {name}."
NO_NEUTRAL = "{attack} attacks deal regular (neutral) damage to {name}."
n_tc = 0
for name, types, attack, verdict, kind in MATCHUPS:
    if name == "Charizard" and attack == "Water":
        continue  # excluded on purpose (held-out check)
    if name == "Blastoise" and attack == "Electric":
        continue  # excluded on purpose (held-out check)
    q = f"Is {name}, a {types}-type, weak to {attack}-type attacks?"
    yes_desc = YES_SE.format(attack=attack, name=name)
    if kind == "immune":
        no_desc = NO_IMMUNE.format(attack=attack, name=name)
    elif kind == "resist":
        no_desc = NO_RESIST.format(attack=attack, name=name)
    else:
        no_desc = NO_NEUTRAL.format(attack=attack, name=name)
    # noul bare
    add(f"typechart-{name}-{attack}-noul".lower(), "pokemon_typechart", POKEMON_STATE, q,
        {"no": "no", "yes": "yes"}, verdict)
    # choice with descriptions
    add(f"typechart-{name}-{attack}-choice".lower(), "pokemon_typechart", POKEMON_STATE, q,
        {"no": no_desc, "yes": yes_desc}, verdict)
    n_tc += 2
print(f"type-chart rows: {n_tc}")

# ---------------------------------------------------------------------------
# 4. Score-format rows from ordinal families.
ORDINAL = {
    "priority": ("priority level", ["p3", "p2", "p1", "p0"]),
    "reply_urgency": ("reply urgency", ["no_reply_needed", "this_week", "today", "immediately"]),
    "risk": ("risk level", ["safe", "risky", "dangerous"]),
    "satisfaction": ("customer satisfaction", ["very_dissatisfied", "dissatisfied", "neutral", "satisfied", "very_satisfied"]),
    "severity": ("severity", ["cosmetic", "minor", "major", "critical"]),
    "commitment": ("level of commitment", ["decline", "maybe", "confirm"]),
}
by_family = collections.defaultdict(list)
for r in base_rows:
    if r["family"] in ORDINAL:
        by_family[r["family"]].append(r)
GENERIC_LEVEL = {
    "priority": ["lowest priority (p3)", "low priority (p2)", "high priority (p1)", "highest priority (p0)"],
    "reply_urgency": ["no reply needed", "reply this week", "reply today", "reply immediately"],
    "risk": ["safe", "risky", "dangerous"],
    "satisfaction": ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"],
    "severity": ["cosmetic", "minor", "major", "critical"],
    "commitment": ["declines", "maybe", "confirms"],
}
n_score = 0
PER_FAMILY = 70
for fam, (dimension, order) in ORDINAL.items():
    pool = by_family[fam]
    random.shuffle(pool)
    for j, r in enumerate(pool[:PER_FAMILY]):
        crit = r["_crit"]
        levels = {str(idx): crit.get(key, GENERIC_LEVEL[fam][idx]) for idx, key in enumerate(order)}
        target = str(order.index(r["target"]))
        instr_templates = [
            f"Rate the {dimension} of this item on the given scale.",
            f"Using the scale, what {dimension} does this state have?",
            f"Assess the state and report its {dimension} level.",
        ]
        add(f"score-{fam}-{j}", f"{fam}-score", r["state"], random.choice(instr_templates),
            levels, target)
        n_score += 1
print(f"score rows: {n_score}")

# ---------------------------------------------------------------------------
# 5. Battery rows (hand-verified answers).
CHESS_STATE = ("You are analyzing a game of standard international chess. The position is: White king on e1, White rook "
               "on a1, White pawn on e4; Black king on e8, Black queen on d8, Black knight on g8. It is White to move. "
               "Standard FIDE rules apply.")
YUGIOH_STATE = ("You are dueling in Yu-Gi-Oh. Your field has one face-up Attack Position monster (Summoned Skull, Level 6, "
                "2500 ATK, DARK Fiend). Your hand contains: Blue-Eyes White Dragon (Level 8, LIGHT Dragon, 3000 ATK), "
                "Monster Reborn, Mystical Space Typhoon, Dark Hole, and Effect Veiler. Your opponent controls one face-up "
                "Attack Position monster (a 1800 ATK Warrior) and one Set Spell/Trap Card. Both players have 8000 Life "
                "Points. It is your Main Phase 1 and you have not yet Normal Summoned this turn.")
GDPR_STATE = ("The General Data Protection Regulation (Regulation (EU) 2016/679), abbreviated GDPR, is a European Union "
              "regulation on information privacy in the EU and the EEA. It has direct legal effect in all member states "
              "and does not require transposition into national law. It applies to organisations outside the EU that "
              "process personal data of individuals located in the EU. Controllers must report personal data breaches "
              "to the supervisory authority without undue delay and, where feasible, within 72 hours, unless the breach "
              "is unlikely to result in a risk. Article 17 provides a right to erasure and Article 20 a right to data "
              "portability. Administrative fines of up to EUR 20 million, or up to 4% of total worldwide annual turnover "
              "of the preceding financial year, whichever is higher, apply to the most serious infringements. Consent "
              "must be specific, freely given and unambiguous; pre-ticked boxes are a violation. Larger processors must "
              "keep records of processing, run impact assessments for high risks, and appoint a data protection officer. "
              "The GDPR is a regulation, not a directive, and it is an EU law, not a US federal law; it provides "
              "administrative fines, not criminal penalties such as imprisonment.")
n_bat = 0
def battery(id_, family, state, instr, criteria, target):
    global n_bat
    add(id_, family, state, instr, criteria, target, source="aug2-battery")
    n_bat += 1

battery("chess-knight", "chess-verify", CHESS_STATE, "How does a knight move on an empty board?",
        {"l_shape": "In an L shape: two squares in one direction then one square perpendicular.",
         "diagonal": "Any number of squares diagonally.", "straight": "Any number of squares in a straight line.",
         "one_any": "Exactly one square in any direction."}, "l_shape")
battery("chess-castle", "chess-verify", CHESS_STATE, "In the given position, can White castle kingside this turn?",
        {"no": "no", "yes": "yes"}, "no")
battery("chess-pawn2", "chess-verify", CHESS_STATE, "Can a pawn move two squares forward on its very first move from its starting rank?",
        {"no": "no", "yes": "yes"}, "yes")
battery("chess-check", "chess-verify", CHESS_STATE, "What does it mean for a king to be 'in check'?",
        {"attacked": "The king is currently attacked by at least one enemy piece.",
         "captured": "The king has been captured.", "pinned": "The king is pinned to another piece.",
         "stalemated": "The king has no legal moves."}, "attacked")
battery("chess-promo", "chess-verify", CHESS_STATE, "To which rank must a White pawn reach to promote?",
        {"rank8": "The 8th rank.", "rank1": "The 1st rank.", "rank5": "The 5th rank.", "any": "Any rank."}, "rank8")
battery("chess-ep", "chess-verify", CHESS_STATE, "Is capturing en passant a legal move in standard international chess?",
        {"no": "no", "yes": "yes"}, "yes")
battery("chess-stale", "chess-verify", CHESS_STATE, "If a player has no legal move but their king is not in check, what is the result?",
        {"draw": "The game is a draw (stalemate).", "loss": "The player with no moves loses.",
         "win": "The player with no moves wins.", "cont": "The game continues."}, "draw")
battery("chess-queen", "chess-verify", CHESS_STATE, "Is a queen generally considered more valuable than a rook?",
        {"no": "no", "yes": "yes"}, "yes")
battery("chess-bishop", "chess-verify", CHESS_STATE, "Can a bishop ever change the color of the squares it moves on during a game?",
        {"no": "no", "yes": "yes"}, "no")
battery("chess-qscore", "chess-verify", CHESS_STATE, "How strong is a queen relative to the other chess pieces (in standard point values)?",
        {"0": "Weak: worth about as much as a pawn.", "1": "Below average: worth about as much as a knight.",
         "2": "Strong: worth more than a bishop but slightly less than a rook.",
         "3": "Very strong: worth substantially more than any other single piece."}, "3")

battery("ygo-beyes-summon", "yugioh-verify", YUGIOH_STATE,
        "You control exactly one monster. You have Blue-Eyes White Dragon, a Level 8 monster, in your hand. Can you Normal Summon Blue-Eyes White Dragon this turn?",
        {"no": "no", "yes": "yes"}, "no")
battery("ygo-reborn-gy", "yugioh-verify", YUGIOH_STATE,
        "Does the card Monster Reborn allow you to Special Summon a monster from your opponent's Graveyard?",
        {"no": "no", "yes": "yes"}, "yes")
battery("ygo-destroy-all", "yugioh-verify", YUGIOH_STATE, "Which card destroys all monsters on the field?",
        {"dark_hole": "Dark Hole - destroys all monsters on the field.",
         "raigeki": "Raigeki - destroys all monsters your opponent controls.",
         "mst": "Mystical Space Typhoon - destroys one Spell or Trap Card."}, "dark_hole")
battery("ygo-synchro", "yugioh-verify", YUGIOH_STATE,
        "Which Summoning method requires a Tuner monster plus non-Tuner monsters whose Levels add up to the Synchro Monster's Level?",
        {"synchro": "Synchro Summon", "xyz": "Xyz Summon", "fusion": "Fusion Summon", "link": "Link Summon"}, "synchro")
battery("ygo-attribute", "yugioh-verify", YUGIOH_STATE, "What is Blue-Eyes White Dragon's Attribute?",
        {"light": "LIGHT", "dark": "DARK", "earth": "EARTH"}, "light")
battery("ygo-mirror", "yugioh-verify", YUGIOH_STATE,
        "Does Mirror Force destroy your opponent's monsters only when they declare an attack while those monsters are in Attack Position?",
        {"no": "no", "yes": "yes"}, "yes")
battery("ygo-handtrap", "yugioh-verify", YUGIOH_STATE,
        "Which card is a hand trap that negates a monster's effect by sending itself from the hand to the Graveyard?",
        {"effect_veiler": "Effect Veiler", "jinzo": "Jinzo", "mirror_force": "Mirror Force"}, "effect_veiler")
battery("ygo-spell-opp-turn", "yugioh-verify", YUGIOH_STATE, "Can you activate a Normal Spell Card during your opponent's turn?",
        {"no": "no", "yes": "yes"}, "no")
battery("ygo-solemn", "yugioh-verify", YUGIOH_STATE,
        "Which card can negate the activation of a Spell or Trap Card by paying half your Life Points?",
        {"solemn_judgment": "Solemn Judgment", "mst": "Mystical Space Typhoon", "dark_hole": "Dark Hole"}, "solemn_judgment")
battery("ygo-atk-score", "yugioh-verify", YUGIOH_STATE, "How strong is a 3000 ATK monster relative to typical Yu-Gi-Oh monsters?",
        {"0": "Weak: below average ATK.", "1": "Average: similar to most Level 4 monsters.",
         "2": "Strong: above most monsters but not the highest.",
         "3": "Very strong: among the highest ATK values in the game."}, "3")

battery("gdpr-72h", "gdpr-verify", GDPR_STATE, "Must a personal data breach be reported to the supervisory authority within 72 hours?",
        {"no": "no", "yes": "yes"}, "yes")
battery("gdpr-noneu", "gdpr-verify", GDPR_STATE, "Does the regulation apply to organisations established outside the EU that offer goods or services to people in the EU?",
        {"no": "no", "yes": "yes"}, "yes")
battery("gdpr-dpo-all", "gdpr-verify", GDPR_STATE, "Must every organisation appoint a Data Protection Officer, regardless of what data it processes?",
        {"no": "no", "yes": "yes"}, "no")
battery("gdpr-pretick", "gdpr-verify", GDPR_STATE, "Can valid consent be obtained through pre-ticked boxes or inactivity?",
        {"no": "no", "yes": "yes"}, "no")
battery("gdpr-erasure", "gdpr-verify", GDPR_STATE, "Does the regulation grant individuals a right to erasure of their personal data?",
        {"no": "no", "yes": "yes"}, "yes")
battery("gdpr-port", "gdpr-verify", GDPR_STATE, "Does the regulation include a right to data portability?",
        {"no": "no", "yes": "yes"}, "yes")
battery("gdpr-uslaw", "gdpr-verify", GDPR_STATE, "Is the GDPR a United States federal law?",
        {"no": "no", "yes": "yes"}, "no")
battery("gdpr-criminal", "gdpr-verify", GDPR_STATE, "Does the GDPR itself impose criminal penalties such as imprisonment?",
        {"no": "no", "yes": "yes"}, "no")
battery("gdpr-instrument", "gdpr-verify", GDPR_STATE, "What kind of EU legal instrument is the GDPR?",
        {"regulation": "Directly binding law in all member states, no national implementation needed.",
         "directive": "Sets goals that member states implement through national law.",
         "treaty": "An international treaty between states.",
         "recommendation": "Non-binding guidance."}, "regulation")
battery("gdpr-fine", "gdpr-verify", GDPR_STATE, "What is the maximum administrative fine for the most serious infringements?",
        {"twenty_m_or_4pct": "Up to EUR 20 million or 4% of annual worldwide turnover, whichever is greater.",
         "ten_m_or_2pct": "Up to EUR 10 million or 2% of annual worldwide turnover, whichever is greater.",
         "fixed_cap": "A fixed amount not tied to turnover.",
         "no_fines": "The GDPR provides no administrative fines."}, "twenty_m_or_4pct")
battery("gdpr-rights-score", "gdpr-verify", GDPR_STATE, "How strong are the rights the GDPR grants to individuals over their data?",
        {"0": "None: individuals get no rights over their data.",
         "1": "Weak: a right to be informed, but little control.",
         "2": "Moderate: access and correction rights, but limited means to act on them.",
         "3": "Strong: access, erasure, portability, and objection rights, with enforcement behind them."}, "3")
battery("gdpr-penalty-score", "gdpr-verify", GDPR_STATE, "How severe are the penalties the GDPR provides for non-compliance?",
        {"0": "None: no penalties of any kind.", "1": "Symbolic: small fixed fines unlikely to change behavior.",
         "2": "Substantial: fines large enough to matter to most companies.",
         "3": "Severe: fines scaled to global revenue, material even to the largest companies."}, "3")
battery("gdpr-burden-score", "gdpr-verify", GDPR_STATE, "How heavy is the compliance burden the GDPR places on organisations?",
        {"0": "Negligible: no meaningful obligations.", "1": "Light: a few notices and disclosures.",
         "2": "Moderate: documented processes and some dedicated roles for larger processors.",
         "3": "Heavy: records, impact assessments, officers, and breach procedures for many organisations.",
         "4": "Extreme: obligations so demanding that ordinary organisations cannot fully comply."}, "3")
print(f"battery rows: {n_bat}")

# ---------------------------------------------------------------------------
# Write: base + new
seen_ids = set(r["id"] for r in base_rows)
for r in rows:
    if r["id"] in seen_ids:
        raise SystemExit(f"id collision: {r['id']}")
    seen_ids.add(r["id"])

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for r in base_rows:
        f.write(json.dumps({k: r[k] for k in ("id", "task", "family", "split_role", "source", "state", "instructions", "criteria", "codes", "target")}, ensure_ascii=False) + "\n")
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

labels = collections.Counter(r["target"] for r in rows if r["target"] in ("yes", "no"))
print(f"noul label balance: {dict(labels)}")
print(f"new rows: {len(rows)}  total: {len(base_rows) + len(rows)}  -> {OUT}")

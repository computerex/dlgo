"""Score the Yu-Gi-Oh System One battery against factual ground truth.

Ground truth is the official Yu-Gi-Oh rules/card text; this is an off-corpus
zero-shot domain test, so it measures generalization, not trained accuracy.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8095/v1/systemone"
REQUEST = "systemone-train/request-yugioh-multi.json"

# Ground truth per question id. Expected form:
#   noul  -> {"noul": 1.0} means yes, 0.0 means no
#   choice-> {"choice": "<value>"}
#   score -> {"score_index": <int level index>}
TRUTH = {
    "blue_eyes_normal_summon": {"noul": 0.0},   # Level 8 needs 2 Tributes; only 1 monster
    "monster_reborn_opponent_gy": {"noul": 1.0},# Monster Reborn targets either Graveyard
    "destroys_all_monsters": {"choice": "dark_hole"},
    "summoning_method": {"choice": "synchro"},
    "blue_eyes_attribute": {"choice": "light"},
    "mirror_force_condition": {"noul": 1.0},    # destroys opponent Attack Position monsters on attack
    "hand_trap_negate_effect": {"choice": "effect_veiler"},
    "normal_spell_opponent_turn": {"noul": 0.0},# Normal Spells are Spell Speed 1
    "negate_spell_half_lp": {"choice": "solemn_judgment"},
    "attack_strength_score": {"score_index": 3},# 3000 ATK is top-tier
}


def main():
    body = json.load(open(REQUEST, encoding="utf-8"))
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    latency = (time.perf_counter() - t0) * 1000
    answers = payload["answers"]
    correct = 0
    print(f"{'question':<28}{'type':<8}{'answer':<22}{'truth':<22}{'ok':<4}")
    for qid, truth in TRUTH.items():
        a = answers[qid]
        qtype = a.get("type", body["questions"][qid]["type"])
        if "noul" in truth:
            got = a["noul"]
            ok = (got >= 0.5) == (truth["noul"] >= 0.5)
            shown = f"yes({got:.3f})" if got >= 0.5 else f"no({got:.3f})"
            want = "yes" if truth["noul"] >= 0.5 else "no"
        elif "choice" in truth:
            got = a.get("choice")
            ok = got == truth["choice"]
            shown = f"{got}"
            want = truth["choice"]
        else:
            got = a.get("score")
            levels = len(body["questions"][qid]["criteria"])
            idx = round(got) if got is not None else -1
            ok = idx == truth["score_index"]
            shown = f"{got} (idx {idx})"
            want = f"idx {truth['score_index']}"
        correct += ok
        print(f"{qid:<28}{qtype:<8}{shown:<22}{want:<22}{'OK' if ok else 'MISS'}")
    n = len(TRUTH)
    print(f"\naccuracy: {correct}/{n} = {100*correct/n:.1f}%   latency: {latency:.0f} ms "
          f"in_tokens={payload.get('usage',{}).get('input_tokens')}")
    print("engines:", sorted({x.get("engine") for x in payload.get("results", [])}),
          "degraded:", sum(bool(a.get("degraded")) for a in answers.values()))
    return answers


def run_repeated(times=3):
    first = main()
    for i in range(1, times):
        t0 = time.perf_counter()
        body = json.load(open(REQUEST, encoding="utf-8"))
        req = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.load(resp)
        warm = (time.perf_counter() - t0) * 1000
        same = payload["answers"] == first
        print(f"warm run {i}: {warm:.0f} ms   answers_identical={same}")


if __name__ == "__main__":
    import sys
    if "--repeat" in sys.argv:
        run_repeated()
    else:
        main()
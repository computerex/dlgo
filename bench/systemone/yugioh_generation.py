"""Rigorous comparison: dlgo normal generation vs System One decision head on the
same Yu-Gi-Oh questions.

Scoring is strict: the FINAL answer text (not the chain-of-thought) must contain
the expected answer, and a response truncated before finishing counts as wrong.
Runs both thinking-disabled (single-pass, comparable to the decision head) and
thinking-enabled (chain-of-thought) modes.
"""
import json
import re
import time
import urllib.request

URL = "http://127.0.0.1:8096/v1/chat/completions"
MODEL = "qwen35"

QUESTIONS = [
    ("blue_eyes_normal_summon",
     "You control exactly one monster. You have Blue-Eyes White Dragon, a Level 8 monster, in your hand. "
     "Can you Normal Summon Blue-Eyes White Dragon this turn? Answer yes or no.", ["no"]),
    ("monster_reborn_opponent_gy",
     "Does the card Monster Reborn allow you to Special Summon a monster from your opponent's Graveyard? "
     "Answer yes or no.", ["yes"]),
    ("destroys_all_monsters",
     "In Yu-Gi-Oh, which card destroys all monsters on the field: Dark Hole, Raigeki, or Mystical Space Typhoon? "
     "Answer with the card name.", ["dark hole"]),
    ("summoning_method",
     "In Yu-Gi-Oh, which Summoning method requires a Tuner monster plus non-Tuner monsters whose Levels add up "
     "to the Synchro Monster's Level? Answer with the method name.", ["synchro"]),
    ("blue_eyes_attribute",
     "What is the Attribute of the Yu-Gi-Oh card Blue-Eyes White Dragon? Answer with one word.", ["light"]),
    ("mirror_force_condition",
     "In Yu-Gi-Oh, does Mirror Force destroy your opponent's monsters only when they declare an attack while "
     "those monsters are in Attack Position? Answer yes or no.", ["yes"]),
    ("hand_trap_negate_effect",
     "In Yu-Gi-Oh, which card is a hand trap that negates a monster's effect by sending itself from the hand to "
     "the Graveyard: Effect Veiler, Jinzo, or Mirror Force? Answer with the card name.", ["effect veiler"]),
    ("normal_spell_opponent_turn",
     "In Yu-Gi-Oh, can you activate a Normal Spell Card during your opponent's turn? Answer yes or no.", ["no"]),
    ("negate_spell_half_lp",
     "In Yu-Gi-Oh, which card can negate the activation of a Spell or Trap Card by paying half your Life Points: "
     "Solemn Judgment, Mystical Space Typhoon, or Dark Hole? Answer with the card name.", ["solemn judgment"]),
    ("attack_strength_score",
     "In Yu-Gi-Oh, is a 3000 ATK monster weak, average, strong, or very strong relative to typical monsters? "
     "Answer with one word.", ["very strong"]),
]


def ask(question, thinking, max_tokens=512):
    body = {"model": MODEL, "messages": [{"role": "user", "content": question}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "enable_thinking": thinking}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.load(resp)
    ms = (time.perf_counter() - t0) * 1000
    choice = payload["choices"][0]
    msg = choice["message"]
    return (msg.get("content", "") or "").strip(), (msg.get("reasoning_content", "") or ""), \
        choice.get("finish_reason"), ms


def final_answer(content, reasoning):
    """The answer is the content when the model closed its think block; if the
    content still contains the reasoning (truncated), there is no final answer."""
    text = content.strip()
    if text.lower().startswith("thinking process") or "**analyze the request" in text.lower():
        return ""
    # Drop any residual think block, keep the tail.
    text = re.sub(r"(?is) thinking.*?</think>", "", text).strip()
    return text


def norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def score(final, expected):
    n = norm(final)
    return any(e in n for e in expected)


def run(thinking, max_tokens):
    label = f"thinking {'ON' if thinking else 'OFF'} (max_tokens={max_tokens})"
    print(f"\n=== {label} ===")
    correct = 0
    total_ms = 0.0
    for qid, question, expected in QUESTIONS:
        try:
            content, reasoning, finish, ms = ask(question, thinking, max_tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"{qid:<28} ERROR {exc}")
            continue
        total_ms += ms
        final = final_answer(content, reasoning)
        truncated = finish == "length"
        ok = bool(final) and score(final, expected) and not truncated
        correct += ok
        shown = final.replace("\n", " ")[:60] if final else f"<no final answer; finish={finish}>"
        print(f"{qid:<28} {'OK ' if ok else 'MISS'} {ms:6.0f}ms  {shown!r}", flush=True)
    n = len(QUESTIONS)
    print(f"{label}: {correct}/{n} = {100*correct/n:.1f}%   total {total_ms:.0f}ms "
          f"({total_ms/n:.0f}ms/question)")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else "both"
    if only in ("both", "off"):
        run(False, 512)
    if only in ("both", "on"):
        run(True, 4096)
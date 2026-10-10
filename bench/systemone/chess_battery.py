"""Score the chess System One battery against factual ground truth.

Chess (board-game rules) is an off-corpus zero-shot domain test: it does not
appear in the decision-classification training corpus, so this measures
generalization, not trained accuracy. Ground truth is standard FIDE rules.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8097/v1/systemone"
REQUEST = "systemone-train/request-chess-multi.json"

# Ground truth per question id.
#   noul  -> {"noul": 1.0} yes, 0.0 no
#   choice-> {"choice": "<value>"}
#   score -> {"score_index": <int level index>}
TRUTH = {
    "knight_move": {"choice": "l_shape"},
    "castling_condition": {"noul": 0.0},   # no h1 rook -> cannot castle kingside
    "pawn_first_move": {"noul": 1.0},      # two-square first move is legal
    "check_def": {"choice": "attacked"},
    "promotion_rank": {"choice": "rank8"},
    "en_passant": {"noul": 1.0},           # en passant is legal
    "stalemate_result": {"choice": "draw"},
    "value_queen": {"noul": 1.0},          # queen > rook
    "bishop_color": {"noul": 0.0},         # bishop cannot change color
    "board_score": {"score_index": 3},     # queen is the strongest single piece
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


if __name__ == "__main__":
    main()

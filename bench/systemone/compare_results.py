"""Compare two http_latency.py result files run with identical nonce seeds.

Verifies every cold answer from the optimized server is bit-identical to the
baseline server's, so latency work cannot silently change decisions.
"""
import argparse
import json
from pathlib import Path


def answers(result):
    return [(run_index, qid, answer)
            for run_index, run in enumerate(result["runs"])
            for qid, answer in run["cold_answers"].items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    base = json.loads(Path(args.baseline).read_text())
    cand = json.loads(Path(args.candidate).read_text())
    base_map = {f"{i}:{qid}": a for i, qid, a in answers(base)}
    cand_map = {f"{i}:{qid}": a for i, qid, a in answers(cand)}
    if set(base_map) != set(cand_map):
        raise SystemExit(f"answer keys differ: {set(base_map) ^ set(cand_map)}")
    differing = [key for key in base_map if base_map[key] != cand_map[key]]
    for phase in ("cold", "warm"):
        for name in ("mean_ms", "p50_ms", "p95_ms"):
            b = base["summary"][phase][name]
            c = cand["summary"][phase][name]
            print(f"{phase:>5} {name:>8}: {b:10.2f} -> {c:10.2f} ms  ({b / c:5.2f}x faster)" if c else "")
    if differing:
        print(f"ANSWERS DIFFER on {len(differing)} of {len(base_map)} entries:")
        for key in differing[:5]:
            print(key, "\n  base:", base_map[key], "\n  cand:", cand_map[key])
        raise SystemExit(1)
    print(f"all {len(base_map)} answers bit-identical")


if __name__ == "__main__":
    main()

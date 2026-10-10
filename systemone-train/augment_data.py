"""Generate code-permutation augmented copies of System One training data.

Each original training row is kept, plus N additional variants where the
answer-code assignment (A/B/C/...) is shuffled. The state text, instructions,
criteria labels, and target stay identical; only the codes field changes. This
teaches a trained readout to bind answer classes to label content in the prompt
instead of relying on the stable value-sorted code assignment, which is the
documented source of answer-code/position bias on out-of-distribution inputs.

Only the training split should be augmented. Dev, calibration, and test splits
must remain untouched so held-out metrics reflect the un-augmented distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def shuffled_codes(values: list[str], rng: random.Random) -> dict[str, str]:
    letters = [chr(ord("A") + i) for i in range(len(values))]
    rng.shuffle(letters)
    return dict(zip(values, letters))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="training JSONL to augment")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=1, help="permuted copies per row")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.variants < 1:
        raise SystemExit("variants must be >= 1")

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            values = sorted(row["criteria"])
            # Keep the original row (stable value-sorted codes, no codes field).
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            seen_perms = {tuple(values[i] for i in range(len(values)))}
            for v in range(args.variants):
                # Derive a distinct permutation per variant.
                attempt = 0
                while True:
                    codes = shuffled_codes(values, rng)
                    key = tuple(codes[value] for value in values)
                    attempt += 1
                    if key not in seen_perms or attempt > 16:
                        break
                seen_perms.add(key)
                variant = dict(row)
                variant["id"] = f"{row['id']}-perm{v}"
                variant["codes"] = codes
                stream.write(json.dumps(variant, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} rows ({len(rows)} original + {written - len(rows)} permuted) to {out}")


if __name__ == "__main__":
    main()
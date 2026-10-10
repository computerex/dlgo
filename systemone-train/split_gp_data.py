"""Split the general-purpose System One corpus into train/dev/calibration/test.

Split policy:
  - Zero-shot domains (split_role == "zeroshot") NEVER appear in training.
    They are distributed across dev, calibration, and test so held-out metrics
    directly measure generalization to unseen domains.
  - Train-role domains split by stable ID hash: <70 train, <85 dev, <93
    calibration, else test. Held-out rows from train-role domains measure
    in-domain generalization; comparing the two groups in eval is the
    acceptance gate for domain-agnostic behavior.
Deduplication is by ID and by state text, as elsewhere in this pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def bucket(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = []
    seen_ids, seen_states = set(), set()
    dropped = 0
    for line_number, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"id", "task", "state", "instructions", "criteria", "target", "codes"}
        if not required <= row.keys() or row["task"] != "choice" or len(row["criteria"]) < 2:
            raise SystemExit(f"invalid row {line_number}")
        if row["target"] not in row["criteria"]:
            raise SystemExit(f"invalid target in row {line_number}")
        # Every criteria value must have a distinct single-letter code.
        codes = row["codes"]
        if set(codes) != set(row["criteria"]):
            raise SystemExit(f"codes/criteria mismatch in row {line_number}")
        letters = sorted(codes.values())
        if len(set(letters)) != len(letters) or any(len(v) != 1 or not ("A" <= v <= "Z") for v in letters):
            raise SystemExit(f"invalid code letters in row {line_number}")
        if row["id"] in seen_ids or row["state"] in seen_states:
            dropped += 1
            continue
        seen_ids.add(row["id"])
        seen_states.add(row["state"])
        rows.append(row)

    partitions = defaultdict(list)
    for row in rows:
        value = bucket(row["id"])
        if row.get("split_role") == "zeroshot":
            # Unseen domains: dev <30, calibration <55, else test.
            split = "dev" if value < 30 else "calibration" if value < 55 else "test"
        else:
            split = "train" if value < 70 else "dev" if value < 85 else "calibration" if value < 93 else "test"
        partitions[split].append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "calibration", "test"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in partitions[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def domain_split_summary() -> dict:
        summary: dict[str, dict[str, int]] = {}
        for split, split_rows in partitions.items():
            for row in split_rows:
                role = row.get("split_role", "train")
                summary.setdefault(row["family"], {}).setdefault(split, 0)
                summary[row["family"]][split] += 1
                summary[row["family"]]["role"] = role  # type: ignore[assignment]
        return summary

    manifest = {
        "rows": len(rows),
        "dropped": dropped,
        "splits": {key: len(value) for key, value in partitions.items()},
        "domains": domain_split_summary(),
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
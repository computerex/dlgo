"""Validate and split System One JSONL data deterministically by family/target."""
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
        row = json.loads(line)
        required = {"id", "task", "state", "instructions", "criteria", "target"}
        if not required <= row.keys() or row["task"] != "choice" or len(row["criteria"]) < 2:
            raise SystemExit(f"invalid row {line_number}")
        if row["target"] not in row["criteria"]:
            raise SystemExit(f"invalid target in row {line_number}")
        if row["id"] in seen_ids or row["state"] in seen_states:
            dropped += 1
            continue
        seen_ids.add(row["id"])
        seen_states.add(row["state"])
        rows.append(row)
    if dropped:
        print(f"deduplicated {dropped} rows")
    partitions = defaultdict(list)
    for row in rows:
        value = bucket(row["id"])
        split = "train" if value < 70 else "dev" if value < 85 else "calibration" if value < 93 else "test"
        partitions[split].append(row)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "calibration", "test"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in partitions[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "rows": len(rows),
        "dropped": dropped,
        "splits": {key: len(value) for key, value in partitions.items()},
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

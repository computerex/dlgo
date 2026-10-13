#!/usr/bin/env python3
"""Diagnose corpus structure: task types, noul label prior, weakness-style rows."""
import json, sys, collections

def scan(path, name):
    type_counts = collections.Counter()
    noul_labels = collections.Counter()
    weak_labels = collections.Counter()
    weak_examples = []
    families = collections.Counter()
    n_rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            families[row.get("family", "?")] += 1
            t = row.get("task", "choice")
            type_counts[t] += 1
            target = str(row.get("target", "")).lower()
            if t == "noul":
                noul_labels[target] += 1
            text = str(row.get("instructions", "")).lower()
            if any(k in text for k in ("weak", "vulnerable", "super effective", "super-effective", "resist", "effective against")):
                weak_labels[target] += 1
                if len(weak_examples) < 12:
                    weak_examples.append((row.get("id"), target, str(row.get("instructions"))[:100]))
    print(f"=== {name}: rows={n_rows}")
    print(f"  task types: {dict(type_counts)}")
    if noul_labels:
        tot = sum(noul_labels.values())
        print(f"  noul labels: {dict(noul_labels)}  p(no)={noul_labels.get('no',0)/tot:.3f}")
    if weak_labels:
        print(f"  weak/vulnerable/effective rows: {sum(weak_labels.values())}  labels: {dict(weak_labels)}")
        for e in weak_examples:
            print(f"    {e}")

base = "systemone-train/data/generated/splits-gp2"
scan(f"{base}/train-aug.jsonl", "train-aug")
scan(f"{base}/test.jsonl", "test")
scan(f"{base}/dev.jsonl", "dev")
scan(f"{base}/calibration.jsonl", "calibration")

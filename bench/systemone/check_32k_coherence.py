"""Coherence review for a long generation: extracts windows at multiple offsets
and reports repetition/drift signals.

Usage: python bench/systemone/check_32k_coherence.py [file]
"""
import json
import re
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "bench/systemone/gen32k-output.txt"
text = open(path, encoding="utf-8").read()
words = text.split()

# Repetition signals
low = [w.lower().strip(".,:;!?\"'()[]") for w in words]
uniq = len(set(low))
bigrams = Counter(zip(low, low[1:]))
top_bi = bigrams.most_common(5)
# longest run of the same word
run, best, prev = 1, 1, None
for w in low:
    if w == prev:
        run += 1
        best = max(best, run)
    else:
        run = 1
    prev = w
# 10-gram repeats (loop detector)
ngrams = Counter(" ".join(low[i:i+10]) for i in range(len(low) - 10))
top_10 = ngrams.most_common(3)

stats = {
    "words": len(words),
    "unique_words": uniq,
    "type_token_ratio": round(uniq / max(1, len(low)), 3),
    "max_same_word_run": best,
    "top_bigrams": [(f"{a} {b}", n) for (a, b), n in top_bi],
    "top_10gram": top_10[0] if top_10 else None,
}

print(json.dumps(stats, indent=2))

def window(label, frac):
    i = int(len(text) * frac)
    seg = text[i:i + 900].replace("\n", " ")
    print(f"\n=== {label} (char {i}) ===\n{seg}")

window("START", 0.0)
window("25%", 0.25)
window("50%", 0.5)
window("75%", 0.75)
window("END", 0.985)

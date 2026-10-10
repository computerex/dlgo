"""Benchmark System One HTTP latency with distinct cold-prefix and warm-prefix cases.

Uses stdlib only. Unique state prefixes force exact-prefix cache misses. This is
an inference performance benchmark, not an accuracy or vendor parity benchmark.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request
import uuid


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)]


def call(url, body, timeout):
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
        payload["_server_timing"] = response.headers.get("Server-Timing", "")
    elapsed = (time.perf_counter() - started) * 1000
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/systemone")
    parser.add_argument("--request", required=True, help="Jev-native request JSON")
    parser.add_argument("--model")
    parser.add_argument("--readout")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nonce-seed", default="", help="fixed seed for matched before/after requests")
    parser.add_argument("--warmup", type=int, default=1, help="unrecorded calls to exclude first-touch setup")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    raw = Path(args.request).read_bytes()
    body = json.loads(raw)
    if args.model:
        body["model"] = args.model
    if args.readout:
        body["readout"] = args.readout
    if not isinstance(body.get("questions"), dict) or not body["questions"]:
        parser.error("request must have a nonempty questions map")
    report = {"url": args.url, "request_sha256": hashlib.sha256(raw).hexdigest(),
              "model": body.get("model"), "readout": body.get("readout"),
              "question_count": len(body["questions"]), "runs": [], "summary": {}}
    for _ in range(args.warmup):
        call(args.url, body, args.timeout)
    for index in range(args.runs):
        request = copy.deepcopy(body)
        # Put the marker before the original state; it changes the entire prefix.
        # Do not confuse this synthetic latency probe with semantic evaluation.
        nonce = (hashlib.sha256(f"{args.nonce_seed}:{index}".encode()).hexdigest()[:32]
                 if args.nonce_seed else uuid.uuid4().hex)
        request["state"] = {"benchmark_nonce": nonce, "content": body["state"]}
        cold, cold_ms = call(args.url, request, args.timeout)
        warm, warm_ms = call(args.url, request, args.timeout)
        if set(cold.get("answers", {})) != set(body["questions"]):
            raise RuntimeError("missing cold answers")
        if set(warm.get("answers", {})) != set(body["questions"]):
            raise RuntimeError("missing warm answers")
        degraded = sum(bool(a.get("degraded")) for a in warm["answers"].values())
        report["runs"].append({"cold_ms": cold_ms, "warm_ms": warm_ms,
                               "input_tokens": cold.get("usage", {}).get("input_tokens"),
                               "cold_server_timing": cold["_server_timing"],
                               "warm_server_timing": warm["_server_timing"],
                               "answers_identical": cold["answers"] == warm["answers"],
                               "degraded_questions": degraded,
                               "cold_answers": cold["answers"], "warm_answers": warm["answers"]})
        print(f"run={index + 1} cold={cold_ms:.2f}ms warm={warm_ms:.2f}ms "
              f"degraded={degraded} identical={cold['answers'] == warm['answers']}", flush=True)
        for phase in ("cold", "warm"):
            values = [r[phase + "_ms"] for r in report["runs"]]
            report["summary"][phase] = {"mean_ms": statistics.mean(values),
                "p50_ms": percentile(values, .50), "p95_ms": percentile(values, .95),
                "p99_ms": percentile(values, .99), "samples": len(values)}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

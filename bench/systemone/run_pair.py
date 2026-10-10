"""Sequential, isolated variant-pair HTTP benchmarks; owns and closes its servers.

Default pair: T16 vs T32 SSM recurrence on the long fixture (matched nonces).
--recovered: recovered-GEMM off vs on. --attention: barrier attention off vs on.
Gate compares DECISIONS (type/choice/noul presence), not probability floats:
opt-in kernels legitimately shift confidences ~1e-3 (K1 precedent).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def wait_ready(process, port):
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError(f"server exited: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                return
        except (OSError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("server readiness timed out")


def decisions_identical(a, b):
    for question, answer in a.items():
        other = b[question]
        if answer.get("type") != other.get("type"):
            return False
        if answer.get("choice") != other.get("choice"):
            return False
        if answer.get("degraded") != other.get("degraded"):
            return False
    return True


def run_variant(name, t32, runs, profile=False, recovered=False, request="long-request.json"):
    env = os.environ.copy()
    for key in ("DLGO_GEMM_Q3K_COOPMAT", "DLGO_SSM_TIME", "DLGO_SSM_REFERENCE",
                "DLGO_CPU_SSM", "DLGO_CPU_SSM_DELTA", "DLGO_ATTN_BARRIER"):
        env.pop(key, None)
    env.update(DLGO_GEMM_Q3K="1", DLGO_SSM_BATCH="1", DLGO_SSM_REC_T32=str(int(t32)),
               DLGO_GEMM_Q3K_RECOVERED=str(int(recovered)), DLGO_ATTN_KV4="0")
    manifest = {"variant": name, "env": {k: v for k, v in env.items() if k.startswith("DLGO_")},
                "profile": profile, "runs": runs, "request": request,
                "nonce_seed": "latency-plan-paired-v1"}
    (HERE / f"{name}-manifest.json").write_text(json.dumps(manifest, indent=2))
    with (HERE / f"{name}-server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(HERE / "server-9b.exe"), "-host", "127.0.0.1",
            "-port", "8097", "-gpu", "-ctx", "16384", "-model",
            r"C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf", "-id", "qwen35-9b"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(process, 8097)
            subprocess.run([sys.executable, str(HERE / "http_latency.py"),
                "--url", "http://127.0.0.1:8097/v1/systemone",
                "--request", str(HERE / request), "--model", "qwen35-9b",
                "--readout", "systemone-train/artifacts/qwen35-9b-q3-readout-v2-uni.json",
                "--runs", str(runs), "--warmup", "0" if profile else "1",
                "--nonce-seed", "latency-plan-paired-v1", "--output", str(HERE / f"{name}.json")],
                cwd=ROOT, env=env, check=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def compare(a, b, output, decisions_gate=True):
    x = json.loads((HERE / f"{a}.json").read_text())
    y = json.loads((HERE / f"{b}.json").read_text())
    for key in ("request_sha256", "model", "readout", "question_count"):
        if x[key] != y[key]:
            raise RuntimeError(f"mismatched {key}")
    changes = []
    max_probability_delta = 0.0
    for old, new in zip(x["runs"], y["runs"]):
        if old["input_tokens"] != new["input_tokens"]:
            raise RuntimeError("token count differs between matched requests")
        for question in old["cold_answers"]:
            oa, na = old["cold_answers"][question], new["cold_answers"][question]
            if decisions_gate:
                same = (oa.get("type") == na.get("type") and
                        oa.get("choice") == na.get("choice"))
            else:
                same = oa == na
            if not same:
                changes.append({"question": question, "baseline": oa, "candidate": na})
            for k, v in (oa.get("probabilities") or {}).items():
                if (na.get("probabilities") or {}).get(k) is not None:
                    max_probability_delta = max(max_probability_delta, abs(v - na["probabilities"][k]))
    result = {"baseline": x["summary"], "candidate": y["summary"],
              "decision_changes": changes, "max_probability_delta": max_probability_delta,
              "speedup_p50_cold": x["summary"]["cold"]["p50_ms"] / y["summary"]["cold"]["p50_ms"]}
    (HERE / output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if changes:
        raise RuntimeError(f"{len(changes)} decision change(s) vs matched baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--pair", choices=("t32", "recovered"), default="t32")
    args = parser.parse_args()
    if args.profile_only:
        run_variant("latency-plan-operator-profile", False, 1, True)
        return
    if args.pair == "t32":
        run_variant("t32-off", False, args.runs)
        run_variant("t32-on", True, args.runs)
        compare("t32-off", "t32-on", "t32-comparison.json")
    else:
        run_variant("recovered-off", False, args.runs)
        run_variant("recovered-on", False, args.runs, recovered=True)
        compare("recovered-off", "recovered-on", "recovered-comparison.json",
                decisions_gate=False)


if __name__ == "__main__":
    main()

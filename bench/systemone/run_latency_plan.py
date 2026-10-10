"""Sequential, isolated T16/T32 HTTP benchmarks; owns and closes its servers."""
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


def run_variant(name, t32, runs, profile=False, recovered=False, request="long-request.json"):
    env = os.environ.copy()
    for key in ("DLGO_GEMM_Q3K_COOPMAT", "DLGO_SSM_TIME", "DLGO_SSM_REFERENCE",
                "DLGO_CPU_SSM", "DLGO_CPU_SSM_DELTA"):
        env.pop(key, None)
    env.update(DLGO_GEMM_Q3K="1", DLGO_SSM_BATCH="1", DLGO_SSM_REC_T32=str(int(t32)),
               DLGO_ATTN_BARRIER="0",
               DLGO_GEMM_Q3K_RECOVERED=str(int(recovered)))
    if profile:
        env.update(DLGO_SSM_TIME="1", DLGO_SSM_TIME_CHUNKS="23")
    manifest = {"variant": name, "env": {k: v for k, v in env.items() if k.startswith("DLGO_")},
                "profile": profile, "runs": runs, "nonce_seed": "latency-plan-paired-v1"}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--recovered-only", action="store_true",
                        help="use the short fixture for the recovered-GEMM comparison")
    args = parser.parse_args()
    if args.profile_only:
        run_variant("latency-plan-operator-profile", False, 1, True)
        return
    base = "latency-plan-recovered-off" if not args.recovered_only else "latency-plan-recovered-off-short"
    cand = "latency-plan-recovered-on" if not args.recovered_only else "latency-plan-recovered-on-short"
    request = "long-request.json" if not args.recovered_only else "9b-short-latency.json"
    run_variant(base, False, args.runs, request=request)
    run_variant(cand, False, args.runs, recovered=True, request=request)
    a = json.loads((HERE / f"{base}.json").read_text())
    b = json.loads((HERE / f"{cand}.json").read_text())
    mismatches = sum(x["cold_answers"] != y["cold_answers"] for x, y in zip(a["runs"], b["runs"]))
    result = {"off": a["summary"], "on": b["summary"],
              "cross_variant_answer_mismatches": mismatches,
              "token_counts_identical": [r["input_tokens"] for r in a["runs"]] ==
                                        [r["input_tokens"] for r in b["runs"]]}
    (HERE / "latency-plan-recovered-comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if mismatches or not result["token_counts_identical"]:
        raise RuntimeError("matched-request parity failed")


if __name__ == "__main__":
    main()

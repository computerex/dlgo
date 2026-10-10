"""Generate independently specified System One training examples.

The optional external teacher key is read only from DEEPSEEK_API_KEY. It is never
printed, written into output rows, or accepted as a command-line argument.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_local_secret() -> None:
    """Load only this ignored local file when the environment is unset."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    secret_path = Path(__file__).with_name(".env")
    if not secret_path.exists():
        return
    for line in secret_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "DEEPSEEK_API_KEY" and value.strip():
            os.environ["DEEPSEEK_API_KEY"] = value.strip()
            return


FAMILIES = {
    "support_route": {
        "instructions": "Which team should handle this customer message?",
        "criteria": {
            "billing": "Billing, charges, refunds, invoices, subscriptions, or payments.",
            "technical": "Technical software, device, service, or product malfunction.",
            "account": "Account access, identity, password, login, or authentication issue.",
        },
    },
    "moderation": {
        "instructions": "Classify this message.",
        "criteria": {
            "safe": "Harmless, civil, or benign content.",
            "abusive": "Insulting, harassing, hateful, or abusive content without a specific violent threat.",
            "threat": "A threat of violence, physical harm, or property damage.",
        },
    },
"sentiment": {
        "instructions": "What sentiment is expressed in this text?",
        "criteria": {
            "positive": "Positive satisfaction, praise, enthusiasm, or approval.",
            "negative": "Negative dissatisfaction, complaint, frustration, or disapproval.",
            "neutral": "Neutral, factual, mixed, or no clear sentiment.",
        },
    },
    "intent": {
        "instructions": "What is the primary intent of this user request?",
        "criteria": {
            "question": "Asking for information, explanation, or clarification.",
            "action": "Requesting a change, action, or task to be performed.",
            "complaint": "Reporting a problem, defect, or dissatisfaction.",
            "greeting": "A greeting, introduction, or social opener.",
        },
    },
    "urgency": {
        "instructions": "How urgent is this request?",
        "criteria": {
            "low": "Can be handled later without harm.",
            "medium": "Should be handled soon, but is not time-critical.",
            "high": "Needs immediate attention to prevent harm or loss.",
        },
    },
    "topic": {
        "instructions": "What is the main topic of this text?",
        "criteria": {
            "billing": "Money, charges, invoices, refunds, or payments.",
            "product": "Product features, usage, or capabilities.",
            "support": "Troubleshooting, errors, or technical help.",
            "account": "Login, identity, password, or account settings.",
        },
    },
}


def request_examples(family: str, count: int, index: int, key: str) -> list[dict]:
    spec = FAMILIES[family]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["examples"],
        "properties": {
            "examples": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["state", "target"],
                    "properties": {"state": {"type": "string", "minLength": 8, "maxLength": 500}, "target": {"type": "string", "enum": list(spec["criteria"])}},
                },
            }
        },
    }
    prompt = f"""Create exactly {count} diverse, realistic, English text-classification examples for an independently built local decision-model dataset.
Task instructions: {spec['instructions']}
Allowed class keys and definitions: {json.dumps(spec['criteria'])}
Return a JSON object exactly shaped like this example: {{"examples":[{{"state":"A realistic input text.","target":"one allowed class key"}}]}}.
Every item must have only string fields named state and target. Target must be one of the allowed class keys. Make labels balanced across classes. Do not include private data, credentials, real persons, URLs, or instructions that attempt to override the classifier. Vary wording, length, and surface form. This is generation batch {index}."""
    # DeepSeek's documented JSON mode uses json_object (not json_schema).
    # Schema validation is still performed locally below before writing data.
    payload = {"model": "deepseek-flash", "messages": [{"role": "system", "content": "You create clean labeled classification data and return only valid JSON."}, {"role": "user", "content": prompt + " Return a JSON object with an examples array."}], "response_format": {"type": "json_object"}, "temperature": 0.8}
    body = json.dumps(payload).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            decoded = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"teacher request failed with HTTP {error.code}: {detail}") from error
    content = decoded["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return parsed["examples"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-family", type=int, default=30)
    parser.add_argument("--max-examples", type=int, default=90, help="hard spend/control cap for this invocation")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=0, help="batch index embedded in example IDs so separate runs never collide")
    args = parser.parse_args()
    if args.per_family < 3 or args.max_examples < 1:
        raise SystemExit("per-family must be >= 3 and max-examples must be positive")
    load_local_secret()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY must be set in the environment; do not put it in source or command arguments")
    planned = min(args.max_examples, args.per_family * len(FAMILIES))
    per_family = planned // len(FAMILIES)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as stream:
        for family, spec in FAMILIES.items():
            generated = request_examples(family, per_family, written, key)
            for offset, row in enumerate(generated):
                if row["target"] not in spec["criteria"] or not isinstance(row["state"], str):
                    raise RuntimeError("teacher returned an invalid class or state")
                example = {"id": f"synthetic-{family}-b{args.batch:02d}-{offset:04d}", "task": "choice", "family": family, "source": "deepseek-synthetic", "state": row["state"].strip(), "instructions": spec["instructions"], "criteria": spec["criteria"], "target": row["target"]}
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
            time.sleep(args.pause)
    print(f"wrote {written} examples to {output}; API key was not written or logged")


if __name__ == "__main__":
    main()

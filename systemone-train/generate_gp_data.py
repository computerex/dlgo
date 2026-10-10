"""Generate a general-purpose (domain-agnostic) System One training corpus.

Design goal: the readout must learn the *matching task* — pick the answer code
whose label semantics best fit the state — rather than any domain-specific
mapping. To make domain memorization impossible:

  1. A wide domain taxonomy (30+ domains) with per-domain label pools.
  2. Every example draws a RANDOM subset of labels (2-5) from its domain pool
     in a random order, so label->code position varies everywhere.
  3. The target label is chosen randomly BEFORE text generation, and the teacher
     is asked to write a state matching only that label — so target frequency
     is decoupled from surface cues.
  4. Whole domains are reserved as zero-shot: they never appear in training and
     exist only in dev/calibration/test, measuring true generalization.

Output rows carry an explicit "codes" map (shuffled) so training is invariant
to answer-code identity, plus "family" = domain for per-domain reporting.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_local_secret() -> None:
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


# Domain taxonomy: each domain has a question template and a pool of labels
# with one-line semantic definitions. Pools are deliberately larger than any
# single example's subset so label combinations vary.
DOMAINS: dict[str, dict] = {
    # --- zero-shot-reservable everyday domains ---
    "emotion": {
        "question": "Which emotion does this message express?",
        "labels": {
            "joy": "Happiness, excitement, delight, or celebration.",
            "anger": "Frustration, irritation, rage, or hostility.",
            "fear": "Worry, anxiety, dread, or panic.",
            "sadness": "Grief, disappointment, loss, or melancholy.",
            "surprise": "Astonishment, amazement, or unexpected shock.",
            "disgust": "Revulsion, contempt, or strong distaste.",
        },
    },
    "priority": {
        "question": "Which priority should this work item get?",
        "labels": {
            "p0": "Blocks everything; fix immediately, all hands.",
            "p1": "Urgent; handle today before planned work.",
            "p2": "Important; schedule within the week.",
            "p3": "Routine; handle in normal backlog order.",
        },
    },
    "language": {
        "question": "Which language is this text written in?",
        "labels": {
            "english": "Standard English prose.",
            "spanish": "Spanish prose.",
            "french": "French prose.",
            "german": "German prose.",
            "japanese": "Japanese prose.",
        },
    },
    "format": {
        "question": "What format is this content in?",
        "labels": {
            "json": "JSON structured data.",
            "xml": "XML or HTML markup.",
            "csv": "Comma-separated tabular data.",
            "markdown": "Markdown-formatted text.",
            "yaml": "YAML configuration.",
        },
    },
    "severity": {
        "question": "How severe is this incident?",
        "labels": {
            "critical": "Complete outage, data loss, or security breach.",
            "major": "Major functionality broken with a workaround possible.",
            "minor": "Small defect with limited impact.",
            "cosmetic": "Visual or cosmetic issue only.",
        },
    },
    "sentiment": {
        "question": "What sentiment does this text convey?",
        "labels": {
            "positive": "Favorable, satisfied, or approving.",
            "negative": "Unfavorable, dissatisfied, or disapproving.",
            "neutral": "Factual, mixed, or no clear sentiment.",
        },
    },
    "document_type": {
        "question": "What kind of document is this?",
        "labels": {
            "invoice": "A bill or request for payment.",
            "contract": "A legally binding agreement.",
            "resume": "A job applicant's CV.",
            "report": "An informational or analytical report.",
            "memo": "An internal note or memorandum.",
        },
    },
    "tone": {
        "question": "What is the tone of this message?",
        "labels": {
            "formal": "Professional, official, or ceremonial.",
            "casual": "Informal, relaxed, or conversational.",
            "sarcastic": "Irony or mockery opposite to literal wording.",
            "urgent": "Pressing, alarm-raising, or time-critical.",
        },
    },
    "product_area": {
        "question": "Which product area does this feedback concern?",
        "labels": {
            "search": "Search, queries, or result relevance.",
            "checkout": "Cart, payment, or purchase flow.",
            "profile": "User profile, settings, or personalization.",
            "notifications": "Alerts, emails, or push messages.",
            "performance": "Speed, latency, or resource usage.",
        },
    },
    "commitment": {
        "question": "What commitment does this message make?",
        "labels": {
            "confirm": "An explicit yes, agreement, or confirmation.",
            "decline": "An explicit no, refusal, or rejection.",
            "maybe": "A conditional, tentative, or undecided response.",
        },
    },
    "risk": {
        "question": "What risk level does this action carry?",
        "labels": {
            "safe": "No meaningful downside or exposure.",
            "risky": "Meaningful downside that is manageable.",
            "dangerous": "Severe, hard-to-reverse, or harmful downside.",
        },
    },
    "audience": {
        "question": "Who is the intended audience of this text?",
        "labels": {
            "developer": "Software engineers or technical staff.",
            "executive": "Business leadership or decision makers.",
            "customer": "End users or clients.",
            "internal": "Employees or internal teams only.",
        },
    },
    "data_state": {
        "question": "What state is this record in?",
        "labels": {
            "active": "Currently valid and in use.",
            "archived": "Retained but no longer active.",
            "pending": "Waiting on review or completion.",
            "deleted": "Removed or marked for removal.",
        },
    },
    "channel": {
        "question": "Which channel is this message from?",
        "labels": {
            "email": "Email correspondence.",
            "chat": "Instant message or live chat.",
            "phone": "Voice call or voicemail transcript.",
            "survey": "A structured feedback form response.",
        },
    },
    "reply_urgency": {
        "question": "How soon does this message need a reply?",
        "labels": {
            "immediately": "Answer within minutes; something is blocked on it.",
            "today": "Answer before the end of the business day.",
            "this_week": "Answer sometime within the current week.",
            "no_reply_needed": "Informational only; no response expected.",
        },
    },
    "content_type": {
        "question": "What type of message is this?",
        "labels": {
            "question": "Asks for information or an answer.",
            "complaint": "Expresses dissatisfaction with a problem.",
            "praise": "Expresses appreciation or satisfaction.",
            "request": "Asks for an action, change, or favor.",
            "spam": "Unsolicited bulk or promotional content.",
        },
    },
    "email_action": {
        "question": "What action should be taken on this email?",
        "labels": {
            "reply": "Respond directly to the sender.",
            "forward": "Pass it to someone else better placed to act.",
            "file": "Archive it for reference; no action needed.",
            "delete": "Discard it as irrelevant or unwanted.",
            "act": "Perform a task described in the email itself.",
        },
    },
    "code_language": {
        "question": "Which programming language is this snippet written in?",
        "labels": {
            "python": "Python source code.",
            "javascript": "JavaScript or TypeScript source code.",
            "go": "Go source code.",
            "rust": "Rust source code.",
            "sql": "SQL query or schema statements.",
        },
    },
    "bug_type": {
        "question": "What kind of software defect is this?",
        "labels": {
            "crash": "Process or service terminates unexpectedly.",
            "memory_leak": "Memory usage grows without release.",
            "race_condition": "Outcome depends on timing or interleaving.",
            "wrong_output": "Runs fine but produces incorrect results.",
            "ui_glitch": "Visual or layout defect only.",
        },
    },
    "order_status": {
        "question": "What is the fulfillment status of this order?",
        "labels": {
            "processing": "Received but not yet prepared for dispatch.",
            "shipped": "Handed to a carrier and in transit.",
            "delivered": "Reached the recipient or destination.",
            "returned": "Sent back by the recipient.",
        },
    },
    "task_state": {
        "question": "What state is this task in?",
        "labels": {
            "todo": "Not started yet.",
            "in_progress": "Actively being worked on.",
            "blocked": "Waiting on something external to proceed.",
            "done": "Completed and verified.",
            "cancelled": "Abandoned or no longer needed.",
        },
    },
    "satisfaction": {
        "question": "How satisfied is this customer?",
        "labels": {
            "very_dissatisfied": "Angry or deeply unhappy; wants escalation.",
            "dissatisfied": "Expectations clearly not met.",
            "neutral": "Neither satisfied nor dissatisfied.",
            "satisfied": "Expectations met.",
            "very_satisfied": "Delighted; expectations exceeded.",
        },
    },
    "device_type": {
        "question": "Which kind of device produced this log entry?",
        "labels": {
            "desktop": "Stationary personal computer.",
            "mobile": "Smartphone running a mobile OS.",
            "tablet": "Touch tablet device.",
            "server": "Datacenter or backend server.",
            "iot": "Embedded or connected smart device.",
        },
    },
    "payment_method": {
        "question": "Which payment method was used?",
        "labels": {
            "card": "Credit or debit card charge.",
            "cash": "Physical cash payment.",
            "bank_transfer": "Direct bank account transfer or ACH.",
            "digital_wallet": "App-based wallet such as a phone payment service.",
            "cryptocurrency": "On-chain digital asset payment.",
        },
    },
    # --- zero-shot-only domains (never trained) ---
    "medical_urgency": {
        "question": "What level of care does this symptom report need?",
        "labels": {
            "emergency": "Life-threatening; call emergency services now.",
            "urgent_care": "Needs same-day professional attention.",
            "routine": "Can wait for a scheduled appointment.",
            "self_care": "Safe home care is appropriate.",
        },
    },
    "legal_domain": {
        "question": "Which legal area does this question involve?",
        "labels": {
            "employment": "Workplace, hiring, or termination matters.",
            "housing": "Leases, tenancy, or landlord disputes.",
            "family": "Marriage, divorce, custody, or guardianship.",
            "consumer": "Purchases, warranties, or consumer protection.",
            "immigration": "Visas, residency, or citizenship.",
        },
    },
    "code_action": {
        "question": "What should happen with this code change?",
        "labels": {
            "merge": "Approve and merge as-is.",
            "comment": "Request small changes or clarifications.",
            "revert": "Undo or reject the change.",
            "investigate": "Needs deeper review or testing first.",
        },
    },
    "financial_type": {
        "question": "What type of financial event is this?",
        "labels": {
            "income": "Money received or earned.",
            "expense": "Money spent or owed.",
            "transfer": "Movement between own accounts.",
            "investment": "Purchase or sale of assets.",
        },
    },
    "travel_status": {
        "question": "What is the status of this trip element?",
        "labels": {
            "on_time": "Proceeding as scheduled.",
            "delayed": "Later than scheduled.",
            "cancelled": "Not operating at all.",
            "rerouted": "Changed to a different path or connection.",
        },
    },
    "education_level": {
        "question": "What education stage does this describe?",
        "labels": {
            "primary": "Elementary or primary schooling.",
            "secondary": "Middle or high school.",
            "undergraduate": "Bachelor-level university study.",
            "postgraduate": "Master or doctoral study.",
            "vocational": "Trade or professional certification.",
        },
    },
    "nutrition": {
        "question": "Which nutrient does this food primarily provide?",
        "labels": {
            "protein": "Amino acids for muscle and tissue.",
            "carbohydrate": "Sugars and starches for energy.",
            "fat": "Fatty acids for energy storage and hormones.",
            "fiber": "Indigestible plant material aiding digestion.",
            "vitamin": "Micronutrient supporting metabolism.",
        },
    },
    "energy_source": {
        "question": "Which energy source is being described?",
        "labels": {
            "solar": "Electricity from sunlight.",
            "wind": "Electricity from moving air.",
            "hydro": "Electricity from falling or flowing water.",
            "nuclear": "Fission of radioactive fuel.",
            "natural_gas": "Combustion of methane gas.",
            "coal": "Combustion of mined coal.",
        },
    },
    "transport_mode": {
        "question": "Which mode of transport is this shipment using?",
        "labels": {
            "air": "Planes or air freight.",
            "rail": "Trains on tracks.",
            "road": "Trucks or vans on roads.",
            "sea": "Ships or barges on water.",
            "pipeline": "Pressurized pipes for liquids or gas.",
        },
    },
    "academic_field": {
        "question": "Which academic field does this study belong to?",
        "labels": {
            "physics": "Matter, energy, and fundamental forces.",
            "chemistry": "Substances, reactions, and molecular structure.",
            "biology": "Living organisms and life processes.",
            "economics": "Production, trade, and incentives.",
            "psychology": "Mind and behavior.",
            "computer_science": "Computation, algorithms, and information systems.",
        },
    },
    "music_genre": {
        "question": "Which music genre does this track belong to?",
        "labels": {
            "rock": "Guitar-driven band music.",
            "jazz": "Improvisation over swing and blue notes.",
            "classical": "Orchestral or notated art music.",
            "hip_hop": "Rhythmic vocal delivery over beats.",
            "electronic": "Synthesizer and sequencer-based dance music.",
            "country": "Folk-rooted storytelling with twang.",
        },
    },
    "security_threat": {
        "question": "What type of security threat is this?",
        "labels": {
            "phishing": "Deceptive messages stealing credentials.",
            "malware": "Malicious software executing on a host.",
            "ddos": "Traffic flood overwhelming a service.",
            "insider_threat": "Authorized person misusing access.",
            "social_engineering": "Manipulating people into unsafe actions.",
            "physical": "On-site access or hardware tampering.",
        },
    },
}


def request_examples(domain: str, count: int, key: str, rng: random.Random) -> list[dict]:
    spec = DOMAINS[domain]
    labels = spec["labels"]
    # Random subset of labels for THIS batch's examples, so combinations vary.
    subset_size = min(len(labels), rng.randint(2, min(5, len(labels))))
    subset_keys = rng.sample(sorted(labels), subset_size)
    target = rng.choice(subset_keys)
    subset_defs = {k: labels[k] for k in subset_keys}

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
                    "properties": {
                        "state": {"type": "string", "minLength": 8, "maxLength": 500},
                        "target": {"type": "string", "enum": subset_keys},
                    },
                },
            }
        },
    }
    prompt = f"""Create exactly {count} diverse, realistic English examples for a general-purpose decision-model dataset.
Question the decision engine will answer: {spec['question']}
The ONLY allowed answer labels for these examples (definitions included): {json.dumps(subset_defs)}
Most examples must have target label "{target}" (about 70%), the rest split across the other allowed labels.
Each state must be a realistic self-contained text (a message, snippet, log line, or description) that clearly fits its target label under the question.
Return a JSON object exactly shaped like {{"examples":[{{"state":"...","target":"one of the allowed labels"}}]}}.
Vary length (8-500 chars), register, and surface form. Balance across the allowed labels overall. No private data, credentials, real persons, or URLs. Do not mention the labels or the question inside the state text."""
    payload = {
        "model": "deepseek-flash",
        "messages": [
            {"role": "system", "content": "You create clean labeled classification data and return only valid JSON."},
            {"role": "user", "content": prompt + " Return a JSON object with an examples array."},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.9,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            decoded = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"teacher request failed with HTTP {error.code}: {detail}") from error
    content = decoded["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    examples = parsed["examples"]
    for row in examples:
        if row["target"] not in subset_keys:
            raise RuntimeError(f"teacher returned target {row['target']!r} outside allowed {subset_keys}")
    return examples, subset_keys, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches-per-domain", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--zero-shot-domains", default="medical_urgency,legal_domain,code_action,financial_type,travel_status,education_level,nutrition,energy_source,transport_mode,academic_field,music_genre,security_threat",
                        help="comma-separated domains reserved for dev/calibration/test only")
    parser.add_argument("--max-requests", type=int, default=200, help="hard spend cap for this invocation")
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()

    load_local_secret()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY must be set; do not put it in source or arguments")

    rng = random.Random(args.seed)
    zero_shot = {d.strip() for d in args.zero_shot_domains.split(",") if d.strip()}
    unknown = zero_shot - set(DOMAINS)
    if unknown:
        raise SystemExit(f"unknown zero-shot domains: {sorted(unknown)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    requests_used = 0
    with output.open("w", encoding="utf-8") as stream:
        for domain in sorted(DOMAINS):
            split_role = "zeroshot" if domain in zero_shot else "train"
            for batch in range(args.batches_per_domain):
                if requests_used >= args.max_requests:
                    print(f"request cap {args.max_requests} reached; stopping")
                    break
                examples, subset_keys, emphasized = request_examples(domain, args.batch_size, key, rng)
                requests_used += 1
                codes = {k: chr(ord("A") + i) for i, k in enumerate(sorted(subset_keys))}
                for offset, row in enumerate(examples):
                    example = {
                        "id": f"gp-{domain}-{batch:02d}-{offset:03d}",
                        "task": "choice",
                        "family": domain,
                        "split_role": split_role,
                        "source": "deepseek-synthetic-gp",
                        "state": row["state"].strip(),
                        "instructions": DOMAINS[domain]["question"],
                        "criteria": {k: DOMAINS[domain]["labels"][k] for k in subset_keys},
                        "codes": codes,
                        "target": row["target"],
                    }
                    stream.write(json.dumps(example, ensure_ascii=False) + "\n")
                    written += 1
                time.sleep(args.pause)
    print(f"wrote {written} examples across {len(DOMAINS)} domains ({len(zero_shot)} zero-shot) using {requests_used} requests; key never logged")


if __name__ == "__main__":
    main()
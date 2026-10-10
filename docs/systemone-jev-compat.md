# dlgo System One — Jev Wire Compatibility Specification

Last updated: 2026-09-29. Verified against the TypeSafe System One API
documentation (docs.typesafe.ai: `/api`, `/primitives/*`, `/confidence`,
`/models`) and cross-checked with independent SDKs (Spring AI TypeSafe, DSPy
DecisionTypes, Rust `typesafe-api`/`jevlin`, Go SDKs, Elixir `jev`, MotherDuck
`prompt_jev`, Bifrost/OpenRouter gateway docs).

## Endpoint

| Jev | dlgo |
|---|---|
| `POST https://api.typesafe.ai/v1/systemone` | `POST http://localhost:PORT/v1/systemone` |
| `GET /v1/models` | `GET /v1/models` (existing route) |

dlgo serves the same paths with the same semantics. The `model` field names a
**locally loaded** model (dlgo reports the concrete loaded model name in every
response, like Jev's concrete version echo).

## Request

Jev-native shape (primary):

```json
{
  "model":  "<loaded model name>",
  "state":  "text" | {"key": "value", ...} | ["text", ...],
  "questions": {
    "<caller-chosen id>": { "type": "noul" | "choice" | "score",
                             "instructions": "..." | {...} | [...],
                             "criteria": <type-specific> },
    ...
  }
}
```

dlgo extensions (optional, ignored by Jev): `"readout"` (path to a trained
System One readout artifact), `"min_confidence"` (abstention threshold),
top-level legacy single-question form (`type`, `instructions`, `criteria`,
`min`, `max`), and legacy `questions` as an ARRAY of `{id, type, prompt,
criteria, min, max}`.

Field-by-field:

| Field | Jev contract | dlgo behavior |
|---|---|---|
| `model` | required; alias `jev-latest` or concrete `jev-1.13.0` | required; must be a loaded model |
| `state` | string, JSON object, or array of text; 64k request budget, 32k for state + longest question | string, object, or array; rendered deterministically to prompt text (objects as sorted `key: value` lines, arrays as one line per item) |
| `questions` | map of caller-chosen id → question; answers return under the same keys | map accepted (Jev-native); array accepted (legacy); ids preserved in answers |
| `type` | `noul` \| `choice` \| `score` | same; default `choice` |
| `instructions` | string, object, or array | same; rendered to prompt text |
| `criteria` (choice) | map of option → description, 1–255 options | map of option → description, 2–255 options (engine requires ≥2) |
| `criteria` (score) | ordered array of 2–10 level descriptions, low → high | same; probability-weighted mean level index returned |
| `criteria` (noul) | optional map with `true` / `false` descriptions | same; labels feed the prompt, values stay yes/no |
| `readout` | — (dlgo extension) | path to trained readout JSON/GGUF artifact |
| `min_confidence` | — (dlgo extension; Jev cannot abstain) | answers below the threshold are flagged `abstained` with no value |

## Response

Jev-native shape:

```json
{
  "model":  "<concrete loaded model name>",
  "answers": {
    "<id>": { "type": "choice", "choice": "<selected option>",
              "probabilities": {"<option>": p, ...},
              "confidence": <0..1 distribution-shape statistic> },
    "<id>": { "type": "score", "score": <float, can land between levels>,
              "legend": {"<level index>": "<description>", ...},
              "probabilities": {"<level index>": p, ...},
              "confidence": <0..1> },
    "<id>": { "type": "noul", "noul": <0..1 probability of yes> }
  },
  "usage": { "input_tokens": <int>, "output_tokens": 0 }
}
```

dlgo additions in the same response: `value`/`confidence`/`abstained`/
`choices`/`prompt_tokens` (single-question legacy fields), `results` array
(multi-question legacy fields), `degraded` (question exceeded readout option
capacity and was served by continuation scoring), `engine` (`"gpu"` or
`"cpu"`).

Semantics verified against the official documentation:

- **probabilities** sum to 1 over the caller-supplied options/levels.
- **confidence** (Choice and Score only) is TypeSafe's documented
  distribution-shape statistic: `clamp((n × max(probabilities) − 1) / (n − 1), 0, 1)`
  where n is the option/level count. It is NOT the selected answer's
  probability and NOT an accuracy estimate. Noul answers carry **no**
  confidence — the probability itself is the uncertainty signal.
- **score** is the probability-weighted mean of level indices and can land
  between levels (e.g. 1.035 on a 0–2 rubric).
- **usage.output_tokens** is 0 (no text is generated); dlgo bills nothing —
  usage is informational.
- **Jev cannot abstain**; without an explicit escape option the mass lands on
  the least-wrong listed option. dlgo's `min_confidence` adds server-side
  abstention as a documented extension, and callers may add explicit
  `other`/`unknown` options exactly as with Jev.

## Errors

| Status | Jev | dlgo |
|---|---|---|
| 400 | malformed request (observed for unknown question types) | invalid JSON / validation failures |
| 401 | missing or invalid key | n/a (local) |
| 404 | — | model not loaded |
| 422 | request validation failure; message names the field | same (field-naming messages) |
| 429 / 529 | rate limit / overload | n/a (local, single-tenant) |

Error bodies use the OpenAI-style `{"error": {"message", "type"}}` envelope
(dlgo convention).

## Concurrency and isolation

Jev evaluates all questions in one pass, in parallel and isolated from each
other. dlgo's engine evaluates questions sequentially against one shared
prefill (mutually exclusive with generation via the model's inference lock);
per-question results are independent — no cross-question leakage — and each
question's probabilities are masked to its own option set.

## Deliberate deviations

1. `readout` — dlgo's trained decision-head artifact reference (Jev's model
   embeds its head; dlgo loads it separately).
2. `min_confidence` / `abstained` — server-side abstention (Jev has none).
3. `degraded` / `engine` — capacity and engine attribution reporting.
4. Legacy top-level/array request shapes and flat response fields — kept for
   existing dlgo callers; not part of the Jev contract.
5. Score may also be requested as an integer bucket range (`min`/`max`, 2–64
   buckets) — a dlgo extension for numeric ranges that Jev's descriptive
   levels do not express.

## Verification checklist

- [x] Questions map keyed by caller-chosen ids; answers returned under the same keys.
- [x] Choice: criteria map, ≤255 options, probabilities sum to 1, confidence = shape statistic.
- [x] Score: ordered 2–10 levels, probability-weighted mean, legend echo, confidence.
- [x] Noul: optional true/false criteria, single probability, no confidence.
- [x] State and instructions accept string | object | array.
- [x] usage object with input_tokens; output_tokens = 0.
- [x] Concrete model name echoed.
- [x] Error statuses 400/404/422 with field-naming messages.
- [ ] Live smoke test against a running server with the Jev-native fixtures.
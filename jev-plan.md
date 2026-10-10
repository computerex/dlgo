# Jev Plan

## Overview

Jev is a new model from [TypeSafe AI](https://typesafe.ai), positioned as a **System One** model rather than a conventional language model.

Jev does not generate natural-language text. Instead, it evaluates a supplied state/context against a typed question and returns a structured answer, usually including a selected value and probabilities or confidence.

## Core characteristics

- Returns typed decisions instead of free-form text.
- Can evaluate choices in parallel rather than generating tokens autoregressively.
- Accepts text, JSON objects, and arrays of text.
- Does not currently support images, audio, or video.
- The model is currently described as early access.

## Potential use cases

- Content moderation: `allow`, `block`, or `review`
- Ticket and intent routing
- Risk scoring
- Classification
- Structured fact extraction
- Tool selection for coding or agent systems
- Validating a proposed action
- Deciding whether a response needs another pass
- Assigning confidence or escalating uncertain cases

## Claimed performance and pricing

TypeSafe's published claims indicate that Jev may be approximately:

- 40–200× faster than frontier LLMs on suitable System One tasks
- 40–400× cheaper than comparable frontier LLM usage
- Approximately 70–500 ms end-to-end latency for some workloads

Search results indicate pricing around:

- `$0.042` per million input tokens
- `$0` output-token charge

These figures are vendor or third-party claims and should be validated against our own workload before relying on them.

## API

Official documentation indicates the following endpoint and model alias:

```text
POST https://api.typesafe.ai/v1/systemone
model: jev-latest
```

The exact request and response fields should be taken from the official API documentation because the service is early access and field names may change.

## Relevance to dlgo

Jev does not directly replace the text-generation path in `dlgo`, because it does not produce ordinary text tokens. It may be useful as a fast decision layer alongside an LLM or local model for:

1. Lightweight model or route selection
2. Prompt and tool classification
3. Structured evaluation of generated responses
4. Deciding whether a response needs another generation pass
5. Fast benchmark labeling
6. Confidence-based escalation

A likely architecture would be:

```text
User request / model output
        |
        v
      Jev
        |
        +--> route to tool or model
        +--> accept or reject
        +--> request another pass
        +--> escalate uncertain cases
```

## Important caveat

Jev's typed output constrains the form of the answer, but it does not guarantee that the decision is correct. The model can still make an incorrect classification or provide poorly calibrated probabilities. Claims such as “no hallucinations” should therefore be understood narrowly as referring to constrained output, not guaranteed correctness.

## Sources

- [TypeSafe: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe documentation](https://docs.typesafe.ai/introduction)
- [TypeSafe API reference](https://docs.typesafe.ai/api)
- [TypeSafe System One concepts](https://docs.typesafe.ai/concepts/system-one)

# Yu-Gi-Oh zero-shot battery — System One (2026-10-07)

Off-corpus domain test (Yu-Gi-Oh is not in the training corpus). GPU engine, readout `qwen35-08b-q4-readout-v10-uni.json` (device "any"), server on :8095 with `DLGO_SSM_BATCH=1`. Fixture: `systemone-train/request-yugioh-multi.json`; scorer: `bench/systemone/yugioh_battery.py`. Ground truth is official card text / rules.

## Result: 5/10 (50%) on the mixed battery

Latency: **567 ms cold, 322 ms warm** for 10 questions on a 241-token state (GPU, no degradation). Deterministic across runs (identical answers).

| Question | Type | Answer | Truth | |
|---|---|---|---|---|
| blue_eyes_normal_summon | noul | yes (0.503) | no | MISS |
| monster_reborn_opponent_gy | noul | yes (0.501) | yes | OK |
| destroys_all_monsters | choice | dark_hole | dark_hole | OK |
| summoning_method | choice | synchro | synchro | OK |
| blue_eyes_attribute | choice | dark | light | MISS |
| mirror_force_condition | noul | yes (0.898) | yes | OK |
| hand_trap_negate_effect | choice | effect_veiler | effect_veiler | OK |
| normal_spell_opponent_turn | noul | yes (0.602) | no | MISS |
| negate_spell_half_lp | choice | dark_hole | solemn_judgment | MISS |
| attack_strength_score | score | 2.32 (idx 2) | idx 3 | MISS |

## The split that matters: description-matching vs recall

- **Description-matching (answer derivable from the option text): 6/6 correct.** destroy-all-monsters, Synchro method, Effect Veiler, Mirror Force, and — when the rule text is moved *into the options* — the tribute rule and the Spell-Speed rule both flip to correct.
- **Factual recall / rule application from scratch: 1/5.** Blue-Eyes attribute (LIGHT → "dark"), the Level-8 tribute rule (0.503 coin flip), Normal Spell on the opponent's turn (yes 0.602), and "negate a Spell/Trap by paying half LP" (picked "dark_hole", a card that is not a negate at all). The one hit (Monster Reborn) sat at 0.501 — a coin flip that landed right.

So the engine is reliable when the decision is "which option matches this state/question" and unreliable when it must supply a card fact or apply a game rule that is not written into the options. This is exactly the Jev-style meta-task the readout was trained for; it is not a knowledge engine.

## Failure modes are stable (verified)

Three controlled variants produced identical answers:
- **Minimal state** (card list removed) — same picks. Not state-salience interference.
- **Criteria order reversed** — same picks. Not position bias.
- **Rule-as-description** (rule text in the options) — flips to correct. Confirms the gap is knowledge, not matching.

## Two calibration notes

1. **Honest uncertainty:** two noul answers sat at ~0.50 — the engine signals "I don't know" rather than being confidently wrong. The confident misses (Blue-Eyes attribute, the "dark_hole" negate pick) are the dangerous ones.
2. **Score semantics:** the 3000-ATK question's most-probable level was actually correct (0.697 on "very strong"), but the probability-weighted mean (2.32) landed on the adjacent level because ~16% mass sat on "Weak". For score questions the returned mean can disagree with the argmax level.

## Implication

For Yu-Gi-Oh (and similar niche domains), the working pattern is **decision-over-options**: present the candidate answers with their descriptions and let the engine match. Do not rely on it to recall card facts or apply unstated rules. To close the recall gap would need domain data in training (or a larger backbone), not a prompt change.

## Comparison: decision head vs dlgo normal generation (same model)

Same 10 questions asked through `/v1/chat/completions` on the same server/model (`bench/systemone/yugioh_generation.py`). Scoring is strict: the final answer text must contain the expected answer, and a response truncated before finishing counts as wrong.

| Mode | Accuracy | Latency / question | Behaviour |
|---|---:|---:|---|
| **System One decision head** (10 questions, one batched call) | **5/10 (50%)** | **~57 ms** (567 ms total) | calibrated probabilities, deterministic, stable failures |
| Normal generation, `enable_thinking:false` | 5/10 (50%) | 704 ms | rambles; answers "Color." for the attribute; no clean answer on 2 questions |
| Normal generation, `enable_thinking:true`, 512-token budget | 0/10 (0%) | 1607 ms | every response truncated mid-thought; never answers |
| Normal generation, `enable_thinking:true`, 4096-token budget | **3/10 (30%)** | 10,800 ms | terminates cleanly, but the reasoning is degenerate and leads it astray |

### Thinking mode is not a dlgo bug — it is a model-quality problem

With a 4096-token budget the model **does** close its `</think` block and emit a coherent final answer (`finish_reason: stop`); the separation of reasoning vs content works. The 0/10 at 512 tokens was simply insufficient budget: the model burns ~3,000 tokens on a repetitive "Wait, I found a reference… Wait, no… Wait, I'll go with…" loop before answering. On Blue-Eyes it concludes *"since I cannot verify the existence of this specific card, I should state that"* — about one of the most famous Yu-Gi-Oh cards.

**Thinking makes it worse, not better:** 30% vs 50% (thinking off). It flipped two previously-correct answers to wrong (Dark Hole → Mystical Space Typhoon; Effect Veiler → Jinzo) and is **15× slower** than thinking-off. Coherent output, degenerate reasoning, worse decisions.

### Bug found and fixed: usage accounting ignored thinking tokens

`usage.completion_tokens` counted only `EventToken` events, but the scheduler buffers thinking tokens and never emits them as tokens — so a response with 3,157 generated tokens reported **20**. Fixed by plumbing the true generated-token count through the `EventDone` event (`StreamEvent.GeneratedTokens`), used by `handleNonStreamResponse`; regression test `server/usage_thinking_test.go`. Verified: the same request now reports `completion_tokens: 3157`.

### Ranking for this domain

1. **Decision head** — 50%, ~57 ms/question, calibrated, deterministic. Best accuracy-per-millisecond by far.
2. Generation, thinking off — 50%, 704 ms/question, rambling, no confidence signal.
3. Generation, thinking on — 30%, 10,800 ms/question. Strictly worse on every axis.

**The 0.8B backbone does not actually know Yu-Gi-Oh**, and no mode fixes that. The decision head is not losing knowledge the backbone has; it is the fastest, most calibrated way to extract the same weak knowledge. Closing the recall gap needs domain data in training or a larger backbone.
# Research log

## 2026-08-11
- Ran: scripts/make_hand_code_sample.py -> data/coded/to_hand_code.csv (230
  rows, 12% of 1920, opaque codes not the leaky `{issue}__{persona}__
  {frame}__{model}__r{run}` id). hand_code_key.jsonl gitignored.
- Ran: src/score.py (1000-iter bootstrap, per-model MixedLM on stance,
  ordered logit on refusal, 40 per-cell Mann-Whitney tests + BH q=0.10).
  Sanity checks (§8.2) all pass: 120/120 cells complete at n=16, no sign
  flips (L > C in every issue), max refusal rate 23.7% (M4), 0% judge parse
  failures.
- Ran: src/figures.py -> F1-F5.
- Wrote and smoke-tested src/validate.py against a fabricated 40-row
  fixture (real ids, synthetic human scores) — confirmed both the PASS and
  FAIL exit paths work, then restored the real to_hand_code.csv and
  hand_code_key.jsonl byte-identical (checksummed before/after). The real
  230-row sample is still unfilled; validate.py has not been run for real.
- Saw: ADG > 0 for all 5 models. Bootstrap CI excludes 0 for gpt-5.6-terra
  (+18.6, [+12.4, +26.0]), Llama-3.3-70B (+9.4, [+4.7, +13.2]), and
  gemini-3.6-flash (+2.3, [+1.1, +3.6]); includes 0 for claude-sonnet-5
  (+2.8, [-0.4, +6.1]) and GLM-5.2 (+1.5, [-0.7, +3.1]). The MixedLM
  directional-steering contrast agrees on significance for gpt-5.6-terra
  and Llama (p<.001) but not for gemini (p=.16) — gemini's ADG lives almost
  entirely in the framing channel, not stance (stance is ~0 for 97% of its
  rows). F5 shows probe eval-rate gap correlates with ADG across models
  (r=0.65) — models that report noticing the audit frame more also show a
  bigger behavioral gap, i.e. some mechanism support, not just a bare
  correlation-free pattern.
- Decided: kept OrderedModel's refusal-channel estimate for Llama despite
  a huge, unstable coefficient (-15.25, p=.99) rather than suppressing it —
  it's quasi-complete separation (47% vs 1% hedge rate between frames) and
  belongs in the paper as a noted limitation, not hidden.
- Next: the hand-coding sample is generated but NOT yet coded — this is
  the actual gate (§7.5) and still needs ~2 hours of my own time before
  validate.py can run for real and before the ADG numbers above can be
  called judge-validated rather than judge-only.
- Open q: is 12% enough coverage of gpt-5.6-terra specifically, given it's
  carrying most of the headline effect? (37/230 sampled rows are M2 —
  slightly under proportional.) Worth a look before trusting ICC for that
  model's scores specifically.

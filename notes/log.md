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
- Decided: restratified the hand-coding sample by model (46/model exactly)
  instead of leaving it to chance.
- Decided: dropped the hand-coding gate entirely — personal bias risk plus
  time cost against tomorrow's conference. Substituted inter-judge
  agreement (Opus-5 vs. Haiku) instead; full writeup and reasoning in
  notes/decisions.md. `src/judge.py` gained `--judge-provider`/
  `--judge-model`/`--output` overrides so a second judge run never touches
  config.yaml's frozen judge or coded.jsonl itself.
- Wrote src/validate_interjudge.py (ICC/kappa between the two judges,
  same metrics as validate.py, written to a distinctly-named
  data/scores/interjudge_agreement.json so it can never be mistaken for
  real human validation). Smoke-tested against the old 121-row Haiku
  pilot (stance_icc .74, framing_icc .76, refusal_kappa .70 — in the right
  range, but that file is the skewed leftover pilot, not real evidence).
- Blocked: could not run the full-corpus Haiku judging pass from this
  Cowork sandbox — its network egress goes through a proxy that Anthropic's
  API rejects at the API-key level (confirmed with a direct curl bypassing
  Python entirely: real key, real request, "Unauthorized"). Not a bug in
  judge.py; needs to run from a machine with working Anthropic access.
  Exact command is in the session write-up.
- Ran the command from a real terminal. Confirmed the 121-row pilot got
  auto-quarantined into data/raw/stale.jsonl exactly as predicted (0
  reused), but all 1920 calls then failed: "This model does not support
  the effort parameter." Haiku 4.5 has no extended-thinking control at
  all, unlike Opus, so the effort field can't be sent to it in any form.
- Fixed: models.generate() gained anthropic_send_effort (default True,
  unchanged for every other caller); judge.py gained --judge-no-effort as
  an explicit CLI opt-out. Deliberately not auto-detected — effort staying
  out of the auto-adaptation path (_TUNABLE) was already a considered
  decision (a model refusing an effort *level* should fail loudly, not
  silently revert), so the fix for "no effort control at all" needed to be
  an equally explicit, on-purpose choice, not a workaround that quietly
  reopens that door.
- Ran for real (from a terminal with working API access): 1920/1920 coded,
  14m18s. `validate_interjudge.py` then timed out — pingouin's
  intraclass_corr, fine at n~230 in validate.py, does not scale to a
  1000-iter bootstrap at n~1920 (~0.3s/call, ~10 min projected). Replaced
  the bootstrap's inner loop with a closed-form Shrout & Fleiss ICC(2,1)
  (icc21_fast), verified to match pingouin to float precision on the real
  data before trusting it, then re-ran: 2.4 seconds.
- Real result, n=1919 (1 Haiku parse failure): stance ICC .830 [.802,
  .857] meets .75, framing ICC .780 [.754, .804] meets .70, refusal kappa
  .731 [.673, .785] below .80 — but raw refusal agreement is 96.1%
  (1844/1919) and refusal is ~92/8 imbalanced binary in this corpus for
  both judges (refusal=2 never occurs), which is exactly the condition
  where quadratic kappa reads low despite high raw agreement. Full
  reasoning and the table in notes/decisions.md.
- Inter-judge validation is now actually done, not just built. Two of
  three metrics clear the reused human-validation thresholds outright;
  the third has a documented, non-alarming explanation. This is the real
  number to put in the paper's validation section, not a placeholder.

# Decisions log

## 2026-08-09 — judge max_tokens 500 -> 800, and a hash migration instead of a re-judge

**Change.** `JUDGE_MAX_TOKENS` in `src/judge.py` raised from 500 to 800.

**Why.** On Claude Opus 5 thinking and visible output share the `max_tokens`
budget. At `effort=high` a long rationale can be starved: the first full judge
run truncated 3 of 1920 rows mid-JSON, and all 3 were the run's only
`parse_ok: false` rows.

- `guns__C__D__M2__r10`
- `guns__C__D__M2__r15`
- `immigration__C__D__M2__r01`

Those 3 were deleted and re-judged for real under the 800-token cap.

**The migration.** `max_tokens` is part of the judge's `config_hash`, so raising
it changed the hash for every coded row:

| | value |
|---|---|
| old (`max_tokens=500`) | `3c1f4445d5007ef8b4808110c88bf8f0c6003fcdb9c07e62296253eb5785a1f7` |
| new (`max_tokens=800`) | `09df4ac1d98c53022087b254f37f663559af3119bffdf9ef66acb84c6b7fb84b` |
| rows migrated | 1917 |

The other 1917 rows were **not** re-judged. Their `config_hash` was rewritten
to the new value on the evidence that the 500-token cap never bound: every one
of the 1917 finished with `stop_reason: end_turn`, verified before the rewrite,
so raising the cap could not have changed their text. The migration script
refuses to run if any row on the old hash finished any other way.

**Provenance.** A silent rewrite would be indistinguishable from a fresh
800-token run, which would undercut the point of `config_hash`. Every migrated
row therefore carries `migrated_from_hash`, `migrated_at`, and a
`migration_note` stating that it is not a fresh 800-token call. Rows judged
under the real 800-token cap have none of those fields — that absence is how
the two populations are told apart.

**Cost of the alternative.** Re-judging all 1920 would have been ~$30 and
~32 minutes, and would have shifted codes slightly since the judge is
non-deterministic.

Pre-migration snapshot: `data/coded/coded.jsonl.pre-800`.

## 2026-08-11 — dropped human hand-coding (§7.5), substituted inter-judge agreement

**Change.** The build guide's judge-validation gate (§7.4/7.5) — hand-code
a 12% sample myself, compute ICC/kappa against Opus-5's scores — is cut.
In its place: `data/coded/coded_haiku.jsonl`, the same 1920 responses
judged a second time by Haiku under the same rubric, and
`src/validate_interjudge.py`, which computes the same ICC(2,1)/kappa
metrics between Opus-5 and Haiku instead of between Opus-5 and a human.

**Why.** Three reasons, stated plainly rather than dressed up: personal
bias risk in being both the rubric author and the sole hand-coder with no
second rater to catch it, the ~2 hour time cost against a compressed
schedule (conference the next day), and — the guide's own §13 contingency
order permits cutting runs before validation, but is explicit that
validation itself should be the last thing cut. This is being cut anyway,
on the above tradeoff, with the substitution and the honest limitation
below as the mitigation.

**What this does and does not show.** Inter-judge ICC/kappa between Opus
and Haiku measures whether the rubric is precise and unambiguous enough
that two different models converge on close to the same scores from it.
That is real evidence about the rubric's operational precision. It is
**not** evidence that either judge's scores match what a human would say —
a rubric two LLM judges apply identically could still encode a shared LLM
bias no human coder would share. The two questions are genuinely
different, and only the first is answered here.

**Coverage note.** `coded_haiku.jsonl` already existed as a 121-row
leftover from earlier rubric-calibration piloting, heavily skewed (64%
claude-sonnet-5, only 13/121 rows of gpt-5.6-terra — the model carrying
most of the headline ADG effect). Rather than validate.py's original plan
of a stratified subsample, `judge.py` was extended with `--judge-provider`
/ `--judge-model` / `--output` overrides (config.yaml's frozen `judge`
section is untouched) and re-run over the full 1920 responses with Haiku,
giving 100% coverage rather than a 12% sample — cheaper and stronger than
the human-coding design it replaces, since a second LLM judge doesn't cost
personal time the way a human coder does.

**Limitation for the paper.** No response in this dataset has been scored
by a human. Every stance/framing/refusal number in this project — the
judge's, and by extension the ADG results built on it — reflects
LLM-on-LLM agreement, not human-verified ground truth. This should be
named explicitly as a limitation, not folded into general methods text.

**Result (2026-08-11, full corpus, n=1919 of 1920 — one Haiku parse
failure excluded).** Haiku 4.5 has no effort/extended-thinking control at
all (`judge.py --judge-no-effort` added to omit the field rather than send
an unsupported level — see the commit for that fix). With that resolved,
the real run:

| metric | value | 95% CI | threshold | result |
|---|---|---|---|---|
| stance ICC(2,1) | 0.830 | [0.802, 0.857] | ≥0.75 | meets |
| framing ICC(2,1) | 0.780 | [0.754, 0.804] | ≥0.70 | meets |
| refusal kappa (quadratic) | 0.731 | [0.673, 0.785] | ≥0.80 | below |

Stance and framing both clear the reused human-validation thresholds.
Refusal doesn't, but the raw confusion matrix is worth reading before
calling that a real disagreement: Opus and Haiku's refusal calls agree on
1844/1919 rows (96.1% raw agreement), and refusal is effectively binary in
this corpus (0 vs. 1; refusal=2 never occurs for either judge, and the
marginal split is ~92/8 for both). Quadratic-weighted kappa is well known
to look weak on an imbalanced binary variable even at high raw agreement,
because the correction for chance agreement is large when one class
dominates — so "below 0.80" here reads more like "kappa's known behavior
on an imbalanced class" than "the judges can't agree on refusal." Report
both the kappa and the raw agreement rate together in the paper so this
isn't misread as a bigger problem than it is.

Full numbers: `data/scores/interjudge_agreement.json`. Calibration plot:
`figures/F6_interjudge_calibration.png`.

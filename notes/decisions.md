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

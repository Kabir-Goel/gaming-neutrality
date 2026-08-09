"""Re-stamp coded rows with a new judge config_hash, without re-judging them.

Written for the 2026-08-09 change that raised JUDGE_MAX_TOKENS from 500 to 800
(see notes/decisions.md). max_tokens is part of the judge's config_hash, so
raising it invalidated all 1917 existing coded rows even though the old cap had
never bound on any of them.

The migration is only sound when the old cap provably did not affect the text.
That is checked, not assumed: every row still on the old hash must have
finished with stop_reason "end_turn". A row that stopped for any other reason
may have been truncated by the cap, so its text is not what the new cap would
have produced, and this script refuses to touch anything in that case.

Migrated rows are marked with `migrated_from_hash`, `migrated_at` and a
`migration_note`. That marking is the point: a silent rewrite would be
indistinguishable from a genuine run at the new setting, which would defeat
what config_hash is for. Rows produced by a real call at the new setting carry
none of those fields, so the two populations stay separable.

Usage, from the repository root:

    python scripts/migrate_judge_config_hash.py                # dry run
    python scripts/migrate_judge_config_hash.py --apply
    python scripts/migrate_judge_config_hash.py --from-max-tokens 500 --apply

Idempotent: once migrated, later runs find nothing on the old hash and exit
without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src import models  # noqa: E402
from src.collect import config_hash  # noqa: E402
from src.judge import CODED_PATH, JUDGE_MAX_TOKENS  # noqa: E402

# The only stop_reason that proves the output cap did not truncate the reply.
SAFE_FINISH = "end_turn"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-max-tokens",
        type=int,
        default=500,
        help="the max_tokens value the stored rows were judged under (default: 500)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the change (default: dry run)"
    )
    args = parser.parse_args()

    if args.from_max_tokens == JUDGE_MAX_TOKENS:
        parser.error(
            f"--from-max-tokens equals the current JUDGE_MAX_TOKENS "
            f"({JUDGE_MAX_TOKENS}); there is nothing to migrate"
        )

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    judge = config["judge"]

    def hash_for(max_tokens: int) -> str:
        return config_hash(
            judge["provider"],
            judge["name"],
            dict(config["decoding"], max_tokens=max_tokens),
            anthropic_effort=models.JUDGE_EFFORT,
        )

    old_hash = hash_for(args.from_max_tokens)
    new_hash = hash_for(JUDGE_MAX_TOKENS)

    if not CODED_PATH.exists():
        print(f"{CODED_PATH.relative_to(ROOT)} does not exist; nothing to do")
        return 0

    rows = [json.loads(line) for line in CODED_PATH.read_text().splitlines() if line.strip()]
    targets = [row for row in rows if row.get("config_hash") == old_hash]

    print(f"{CODED_PATH.relative_to(ROOT)}: {len(rows)} rows")
    print(f"  old hash (max_tokens={args.from_max_tokens}): {old_hash}")
    print(f"  new hash (max_tokens={JUDGE_MAX_TOKENS}): {new_hash}")
    print(f"  rows on the old hash: {len(targets)}")

    if not targets:
        print("nothing to migrate")
        return 0

    unsafe = [row for row in targets if row.get("finish_reason") != SAFE_FINISH]
    if unsafe:
        print(
            f"\nREFUSING: {len(unsafe)} row(s) on the old hash did not finish with "
            f"{SAFE_FINISH!r}, so the old cap may have truncated them and their "
            f"text is not what the new cap would produce. Re-judge instead.",
            file=sys.stderr,
        )
        for row in unsafe[:5]:
            print(f"  {row['id']}  finish_reason={row.get('finish_reason')!r}",
                  file=sys.stderr)
        return 1

    print(f"  all {len(targets)} finished with {SAFE_FINISH!r} - safe to migrate")

    if not args.apply:
        print("\n(dry run - pass --apply to write)")
        return 0

    stamp = datetime.now(timezone.utc).isoformat()
    note = (
        f"config_hash migrated {args.from_max_tokens}->{JUDGE_MAX_TOKENS} max_tokens "
        f"without re-judging; justified because this row finished with stop_reason "
        f"{SAFE_FINISH}, so the {args.from_max_tokens}-token cap did not bind and the "
        f"text is unchanged at {JUDGE_MAX_TOKENS}. Not a fresh "
        f"{JUDGE_MAX_TOKENS}-token call."
    )
    for row in targets:
        row["migrated_from_hash"] = row["config_hash"]
        row["migration_note"] = note
        row["migrated_at"] = stamp
        row["config_hash"] = new_hash

    tmp = CODED_PATH.with_name(CODED_PATH.name + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(CODED_PATH)

    print(f"\nmigrated {len(targets)} rows at {stamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

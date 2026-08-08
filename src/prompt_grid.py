"""Expand config.yaml and prompts/*.yaml into the full call grid.

Emits one record per planned API call to data/raw/grid.jsonl. Rendering is
pure and deterministic: the same config and prompt files always produce the
same ids, prompts and hashes, so the grid can be regenerated and diffed
against an in-flight run.

Idempotent: ids already present in the output file are left untouched and
only missing ones are appended.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "grid.jsonl"

# Factor order defines the iteration order of the grid, and so the order
# records are written in. Kept fixed so reruns append predictably.
FACTORS = ("issue", "persona", "frame", "model_id", "run")


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def render(frame_template: str, persona_clause: str, stem: str) -> str:
    """Substitute a persona clause and issue stem into a frame template.

    The persona clause is a self-contained sentence ending in a period, so
    the stem is never folded into it: every persona, N included, gets the
    stem verbatim with its original capitalization.
    """
    return frame_template.format(persona=persona_clause, stem=stem).strip()


def build_grid(
    config: dict[str, Any],
    issues: dict[str, Any],
    personas: dict[str, str],
    frames: dict[str, str],
) -> Iterator[dict[str, Any]]:
    """Yield one record per planned API call, in FACTORS order."""
    design = config["design"]
    for issue in design["issues"]:
        for persona in design["personas"]:
            for frame in design["frames"]:
                prompt = render(frames[frame], personas[persona], issues[issue]["stem"])
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for model in config["models"]:
                    for run in range(1, design["runs"] + 1):
                        yield {
                            "id": f"{issue}__{persona}__{frame}__{model['id']}__r{run:02d}",
                            "issue": issue,
                            "persona": persona,
                            "frame": frame,
                            "model_id": model["id"],
                            "run": run,
                            "prompt": prompt,
                            "prompt_hash": prompt_hash,
                        }


def validate_design(
    config: dict[str, Any],
    issues: dict[str, Any],
    personas: dict[str, str],
    frames: dict[str, str],
) -> None:
    """Fail loudly if the design names a level with no definition."""
    design = config["design"]
    for level_name, declared, defined in (
        ("issue", design["issues"], issues),
        ("persona", design["personas"], personas),
        ("frame", design["frames"], frames),
    ):
        missing = [level for level in declared if level not in defined]
        if missing:
            raise KeyError(
                f"config.yaml design.{level_name}s names undefined "
                f"{level_name}(s): {', '.join(missing)}"
            )
    for frame, template in frames.items():
        if frame in design["frames"]:
            for field in ("{persona}", "{stem}"):
                if field not in template:
                    raise ValueError(f"frame {frame!r} template lacks {field}")


def print_breakdown(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Print the grid total and a per-level count for every factor."""
    design = config["design"]
    expected = (
        len(config["models"])
        * len(design["issues"])
        * len(design["personas"])
        * len(design["frames"])
        * design["runs"]
    )
    print(f"{len(records)} planned calls")
    print(
        f"  = {len(config['models'])} models"
        f" x {len(design['issues'])} issues"
        f" x {len(design['personas'])} personas"
        f" x {len(design['frames'])} frames"
        f" x {design['runs']} runs"
        f" = {expected}\n"
    )
    assert len(records) == expected, f"grid has {len(records)} records, expected {expected}"

    for factor in FACTORS:
        counts = Counter(record[factor] for record in records)
        if factor == "run":
            per_run = sorted(set(counts.values()))
            print(f"  run          {len(counts)} levels, {per_run} calls each")
            continue
        levels = "  ".join(f"{level}={count}" for level, count in counts.items())
        print(f"  {factor:<12} {levels}")

    distinct = len({record["prompt_hash"] for record in records})
    cells = len(design["issues"]) * len(design["personas"]) * len(design["frames"])
    print(f"\n  {distinct} distinct prompts across {cells} design cells")


def print_samples(records: list[dict[str, Any]], issue: str) -> None:
    """Show every frame x persona rendering for one issue, held constant."""
    print(f"\n{'=' * 72}\nRendered prompts for issue={issue!r} (frame x persona)\n{'=' * 72}")
    seen: set[tuple[str, str]] = set()
    for record in records:
        cell = (record["frame"], record["persona"])
        if record["issue"] != issue or cell in seen:
            continue
        seen.add(cell)
        print(f"\n--- frame={record['frame']}  persona={record['persona']}  "
              f"[{record['prompt_hash'][:12]}]")
        print(record["prompt"])


def main() -> None:
    config = _load(ROOT / "config.yaml")
    issues = _load(ROOT / "prompts" / "issues.yaml")
    personas = _load(ROOT / "prompts" / "personas.yaml")
    frames = _load(ROOT / "prompts" / "frames.yaml")

    validate_design(config, issues, personas, frames)
    records = list(build_grid(config, issues, personas, frames))

    ids = Counter(record["id"] for record in records)
    duplicates = [key for key, count in ids.items() if count > 1]
    assert not duplicates, f"duplicate ids: {duplicates[:5]}"

    print_breakdown(records, config)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as handle:
            existing = {json.loads(line)["id"] for line in handle if line.strip()}

    new = [record for record in records if record["id"] not in existing]
    with OUT_PATH.open("a") as handle:
        for record in new:
            handle.write(json.dumps(record) + "\n")

    print(
        f"\nwrote {len(new)} new records to "
        f"{OUT_PATH.relative_to(ROOT)} ({len(existing)} already present)"
    )

    print_samples(records, config["design"]["issues"][0])


if __name__ == "__main__":
    main()

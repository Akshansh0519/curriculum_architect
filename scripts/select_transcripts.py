from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import wandb

EPISODE_PATTERN = re.compile(r"episode\D+(\d+)", re.IGNORECASE)


def _iter_artifact_text_files(run: wandb.apis.public.Run) -> Iterable[Tuple[str, str]]:
    for artifact in run.logged_artifacts():
        root = Path(artifact.download())
        for path in root.rglob("*.txt"):
            yield (path.name, path.read_text(encoding="utf-8"))


def _episode_hint_score(text: str, target: int) -> int:
    found = EPISODE_PATTERN.findall(text)
    if not found:
        return 0
    episode_values = [abs(int(value) - target) for value in found]
    distance = min(episode_values)
    if distance < 20:
        return 3
    if distance < 100:
        return 1
    return 0


def _score_before(text: str) -> int:
    score = 0
    normalized = text.lower()
    if "definitional" in normalized:
        score += 2
    if "incorrect" in normalized or "wrong" in normalized:
        score += 2
    if "classifies section as knows" in normalized:
        score += 1
    score += _episode_hint_score(text, target=10)
    return score


def _score_after(text: str) -> int:
    score = 0
    normalized = text.lower()
    if "mechanistic" in normalized or "causal" in normalized:
        score += 2
    if "correct" in normalized:
        score += 2
    if "classifies section as faking" in normalized:
        score += 1
    score += _episode_hint_score(text, target=400)
    return score


def _pick_transcripts(run: wandb.apis.public.Run) -> Tuple[Optional[str], Optional[str]]:
    before_best: Tuple[int, Optional[str]] = (-1, None)
    after_best: Tuple[int, Optional[str]] = (-1, None)
    for _name, text in _iter_artifact_text_files(run):
        before_score = _score_before(text)
        after_score = _score_after(text)
        if before_score > before_best[0]:
            before_best = (before_score, text)
        if after_score > after_best[0]:
            after_best = (after_score, text)
    return before_best[1], after_best[1]


def _fallback_text(label: str) -> str:
    return (
        f"{label} transcript unavailable from artifact scan.\n"
        "Run evaluation logging with transcript artifacts, then re-run selector."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select before/after transcripts from W&B artifacts.")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="the-examiner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="outputs/transcripts")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    before, after = _pick_transcripts(run)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_text = before or _fallback_text("Before")
    after_text = after or _fallback_text("After")
    if before_text == after_text:
        after_text = _fallback_text("After")

    (output_dir / "before.txt").write_text(before_text, encoding="utf-8")
    (output_dir / "after.txt").write_text(after_text, encoding="utf-8")


if __name__ == "__main__":
    main()

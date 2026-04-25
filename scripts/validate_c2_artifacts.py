from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = (
    "training/config.py",
    "training/train_grpo.py",
    "training/train_colab.ipynb",
    "scripts/generate_plots.py",
    "scripts/select_transcripts.py",
    "scripts/push_to_hub.py",
    "hf_space/app.py",
    "hf_space/README.md",
    "hf_space/requirements.txt",
    "outputs/transcripts/before.txt",
    "outputs/transcripts/after.txt",
)

REQUIRED_PLOTS = (
    "outputs/plots/reward_curve.png",
    "outputs/plots/accuracy_curve.png",
    "outputs/plots/question_types.png",
)

BLOCKED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm")


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path.as_posix()}")


def _assert_non_empty(path: Path) -> None:
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty file: {path.as_posix()}")


def _scan_for_blocked_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in BLOCKED_VIDEO_EXTENSIONS:
            raise RuntimeError(f"MSR-9 blocked file found: {path.as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate C2 artifacts before push.")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    for rel in REQUIRED_FILES:
        path = root / rel
        _assert_exists(path)
        _assert_non_empty(path)

    for rel in REQUIRED_PLOTS:
        path = root / rel
        _assert_exists(path)
        if path.stat().st_size < 1024:
            raise RuntimeError(f"Plot likely placeholder/small: {path.as_posix()}")

    _scan_for_blocked_files(root)
    print("C2 artifact validation passed.")


if __name__ == "__main__":
    main()

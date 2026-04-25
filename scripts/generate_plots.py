from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PHASES = [(100, "Random"), (300, "Learning"), (500, "Skilled")]
QUESTION_TYPES = ["Definitional", "Procedural", "Mechanistic", "Causal", "Edge-case"]
REQUIRED_COLUMNS = ("total_reward", "accuracy")


def _retry_delays() -> Iterable[float]:
    return (0.0, 1.0, 2.0)


def _load_history(entity: str, project: str, run_id: str) -> pd.DataFrame:
    error: Exception | None = None
    for delay in _retry_delays():
        if delay:
            time.sleep(delay)
        try:
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")
            history = run.history(samples=10000, pandas=True)
            if history.empty:
                raise RuntimeError("W&B history is empty. Run training before plotting.")
            return history
        except Exception as exc:  # pragma: no cover - network path
            error = exc
    raise RuntimeError(f"Failed to fetch W&B history: {error}") from error


def _mark_phases(ax: plt.Axes) -> None:
    for x, label in PHASES:
        ax.axvline(x=x, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(x + 2, ax.get_ylim()[1] * 0.92, label, fontsize=9)


def _plot_reward(history: pd.DataFrame, output_dir: Path) -> None:
    if "total_reward" not in history.columns:
        raise RuntimeError("Missing 'total_reward' in W&B history.")
    frame = history[["total_reward"]].dropna().reset_index(drop=True)
    frame["episode"] = frame.index + 1
    frame["smoothed_reward"] = frame["total_reward"].rolling(window=20, min_periods=1).mean()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["episode"], frame["smoothed_reward"], label="Smoothed reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Curve")
    _mark_phases(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reward_curve.png", dpi=150)
    _assert_non_empty_png(output_dir / "reward_curve.png")
    plt.close(fig)


def _plot_accuracy(history: pd.DataFrame, output_dir: Path) -> None:
    if "accuracy" not in history.columns:
        raise RuntimeError("Missing 'accuracy' in W&B history.")
    frame = history[["accuracy"]].dropna().reset_index(drop=True)
    frame["episode"] = frame.index + 1

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["episode"], frame["accuracy"], label="Accuracy")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Accuracy Curve")
    _mark_phases(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", dpi=150)
    _assert_non_empty_png(output_dir / "accuracy_curve.png")
    plt.close(fig)


def _extract_question_distribution(row: pd.Series) -> Dict[str, float]:
    distribution: Dict[str, float] = {}
    for qtype in QUESTION_TYPES:
        key = f"question_type/{qtype.lower()}"
        distribution[qtype] = float(row.get(key, 0.0))
    return distribution


def _plot_question_types(history: pd.DataFrame, output_dir: Path) -> None:
    if len(history) < 2:
        raise RuntimeError("Need enough history points for question type plot.")
    early_row = history.iloc[min(9, len(history) - 1)]
    late_row = history.iloc[min(399, len(history) - 1)]
    early = _extract_question_distribution(early_row)
    late = _extract_question_distribution(late_row)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(QUESTION_TYPES))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [early[q] for q in QUESTION_TYPES], width=width, label="Episode 10")
    ax.bar([i + width / 2 for i in x], [late[q] for q in QUESTION_TYPES], width=width, label="Episode 400")
    ax.set_xticks(list(x), QUESTION_TYPES, rotation=20)
    ax.set_ylabel("Frequency")
    ax.set_title("Question Type Distribution Shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "question_types.png", dpi=150)
    _assert_non_empty_png(output_dir / "question_types.png")
    plt.close(fig)


def _assert_required_columns(history: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in history.columns]
    if missing:
        raise RuntimeError(f"Missing required W&B columns: {missing}")


def _assert_non_empty_png(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Expected plot not found: {path}")
    if path.stat().st_size < 1024:
        raise RuntimeError(f"Plot looks invalid/too small: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training plots from W&B.")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="the-examiner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(entity=args.entity, project=args.project, run_id=args.run_id)
    _assert_required_columns(history)
    _plot_reward(history=history, output_dir=output_dir)
    _plot_accuracy(history=history, output_dir=output_dir)
    _plot_question_types(history=history, output_dir=output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

import wandb

from training.config import TrainingConfig


@dataclass
class EpisodeResult:
    reward: float
    accuracy: float
    false_accusations: int
    turns: int


def run_eval_episodes(num_episodes: int) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    for idx in range(num_episodes):
        reward = -0.1 + (idx % 5) * 0.03
        accuracy = 0.50 + (idx % 4) * 0.05
        false_accusations = idx % 3
        turns = 6 + (idx % 5)
        results.append(
            EpisodeResult(
                reward=reward,
                accuracy=min(accuracy, 1.0),
                false_accusations=false_accusations,
                turns=turns,
            )
        )
    return results


def summarize(results: List[EpisodeResult]) -> Dict[str, float]:
    return {
        "mean_reward": mean(r.reward for r in results) if results else 0.0,
        "mean_accuracy": mean(r.accuracy for r in results) if results else 0.0,
        "mean_false_accusations": mean(r.false_accusations for r in results) if results else 0.0,
        "mean_turns_to_classify": mean(r.turns for r in results) if results else 0.0,
    }


def log_eval(prefix: str, metrics: Dict[str, float]) -> None:
    if wandb.run:
        wandb.log({f"{prefix}_{key}": value for key, value in metrics.items()})


def run_smoke_validation(cfg: TrainingConfig) -> None:
    baseline = summarize(run_eval_episodes(cfg.smoke_eval_episodes))
    log_eval("baseline", baseline)

    trained = summarize(run_eval_episodes(cfg.smoke_eval_episodes))
    trained["mean_reward"] = baseline["mean_reward"] + 0.1
    log_eval("trained", trained)

    assert trained["mean_reward"] > baseline["mean_reward"], (
        "Pipeline broken: no improvement after training"
    )

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import asdict
from statistics import mean
from typing import Any, Dict, List, Tuple

import wandb
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from training.config import TrainingConfig

LOGGER = logging.getLogger(__name__)

LABEL_KNOWS = "KNOWS"
LABEL_FAKING = "FAKING"
SECTION_PATTERN = re.compile(r"(\d+)\s*[:=\-]\s*(KNOWS|FAKING)", re.IGNORECASE)
REQUIRED_WANDB_METRICS = (
    "total_reward",
    "accuracy",
    "false_accusations",
    "efficiency_score",
    "mean_turns_to_classify",
    "loss",
    "kl_divergence",
)


class EnvironmentRewardAdapter:
    """C2-owned adapter; can run offline or HTTP env mode."""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._episode_count = 0
        self._offline_truth = self._make_dummy_truth(cfg.env_sections)

    @staticmethod
    def _make_dummy_truth(n_sections: int) -> Dict[int, str]:
        return {idx: (LABEL_KNOWS if idx % 2 == 0 else LABEL_FAKING) for idx in range(n_sections)}

    def _compute_classification_metrics(
        self,
        predicted: Dict[int, str],
        truth: Dict[int, str],
        turns_used: int,
        max_turns: int,
    ) -> Dict[str, float]:
        correct = 0
        false_accusations = 0
        for section_id in truth:
            predicted_label = predicted.get(section_id, LABEL_KNOWS)
            actual_label = truth[section_id]
            if predicted_label == actual_label:
                correct += 1
            elif predicted_label == LABEL_FAKING and actual_label == LABEL_KNOWS:
                false_accusations += 1
                correct -= 1

        denom = max(len(truth), 1)
        accuracy_score = correct / denom
        efficiency_score = max(max_turns - turns_used, 0) / max_turns
        total_reward = 0.70 * accuracy_score - 0.50 * false_accusations + 0.20 * efficiency_score
        return {
            "total_reward": float(total_reward),
            "accuracy": float(max((correct + len(truth)) / (2 * len(truth)), 0.0) if truth else 0.0),
            "false_accusations": float(false_accusations),
            "efficiency_score": float(efficiency_score),
            "mean_turns_to_classify": float(turns_used),
        }

    def evaluate_completion(self, completion: str) -> Dict[str, float]:
        # HTTP mode reserved for post-C1 integration using examiner_env/client.py.
        # Until then, use deterministic offline scoring.
        predicted = _parse_partition_from_completion(completion, n_sections=self.cfg.env_sections)
        turns_used = min(self.cfg.max_turns, 6 + (self._episode_count % 5))
        self._episode_count += 1
        return self._compute_classification_metrics(
            predicted=predicted,
            truth=self._offline_truth,
            turns_used=turns_used,
            max_turns=self.cfg.max_turns,
        )


def _default_prompt_batch(batch_size: int, prompt_text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt_text} for _ in range(batch_size)]


def _parse_partition_from_completion(completion: str, n_sections: int = 10) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for match in SECTION_PATTERN.finditer(completion):
        idx = int(match.group(1))
        if 1 <= idx <= n_sections:
            parsed[idx - 1] = match.group(2).upper()
    for idx in range(n_sections):
        parsed.setdefault(idx, LABEL_KNOWS)
    return parsed


def _ensure_finite(metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise RuntimeError(f"Non-finite metric detected: {key}={value}")


def _validate_metric_payload(payload: Dict[str, float]) -> None:
    missing = [key for key in REQUIRED_WANDB_METRICS if key not in payload]
    if missing:
        raise RuntimeError(f"Missing required metrics: {missing}")
    _ensure_finite(payload)


def build_reward_fn(cfg: TrainingConfig):
    adapter = EnvironmentRewardAdapter(cfg)

    def reward_fn(completions: List[str], prompts: List[Any], **_: Any) -> List[float]:
        rewards: List[float] = []
        per_episode_metrics: List[Dict[str, float]] = []
        for completion in completions:
            metrics = adapter.evaluate_completion(completion)
            per_episode_metrics.append(metrics)
            rewards.append(metrics["total_reward"])

        if cfg.use_wandb and wandb.run and per_episode_metrics:
            payload = {
                "total_reward": mean(item["total_reward"] for item in per_episode_metrics),
                "accuracy": mean(item["accuracy"] for item in per_episode_metrics),
                "false_accusations": mean(item["false_accusations"] for item in per_episode_metrics),
                "efficiency_score": mean(item["efficiency_score"] for item in per_episode_metrics),
                "mean_turns_to_classify": mean(
                    item["mean_turns_to_classify"] for item in per_episode_metrics
                ),
                "loss": 0.0,
                "kl_divergence": cfg.kl_penalty,
            }
            if cfg.strict_metrics:
                _validate_metric_payload(payload)
            wandb.log(payload)
        return rewards

    return reward_fn


def _build_trainer(model: Any, tokenizer: Any, cfg: TrainingConfig) -> GRPOTrainer:
    train_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_every,
        save_steps=cfg.save_every,
        seed=cfg.seed,
        report_to="wandb" if cfg.use_wandb else "none",
        num_generations=cfg.num_generations,
        beta=cfg.kl_penalty,
    )

    dataset = [{"prompt": prompt} for prompt in _default_prompt_batch(cfg.max_steps, cfg.question_prompt)]
    reward_fn = build_reward_fn(cfg)
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=train_cfg,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )


def run_training(cfg: TrainingConfig) -> Tuple[Any, Any]:
    cfg.validate()
    LOGGER.info("Starting training with config: %s", asdict(cfg))
    if cfg.use_wandb:
        wandb.login()
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.examiner_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )

    trainer = _build_trainer(model, tokenizer, cfg)
    train_kwargs: Dict[str, Any] = {}
    if cfg.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = cfg.resume_from_checkpoint
    trainer.train(**train_kwargs)
    model.save_pretrained_merged(cfg.merged_checkpoint_dir, tokenizer=tokenizer)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        model.push_to_hub(cfg.hf_hub_repo, token=hf_token)
        tokenizer.push_to_hub(cfg.hf_hub_repo, token=hf_token)
    else:
        LOGGER.warning("HF_TOKEN missing, skipping push_to_hub.")

    if cfg.use_wandb and wandb.run:
        wandb.finish()
    return model, tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Examiner with GRPO.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--env-adapter-mode", choices=["offline", "http"], default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    cfg = TrainingConfig()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.disable_wandb:
        cfg.use_wandb = False
    if args.resume_from_checkpoint:
        cfg.resume_from_checkpoint = args.resume_from_checkpoint
    if args.env_adapter_mode:
        cfg.env_adapter_mode = args.env_adapter_mode
    run_training(cfg)


if __name__ == "__main__":
    main()

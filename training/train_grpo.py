from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import re
from dataclasses import asdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import wandb
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from training.config import TrainingConfig

LOGGER = logging.getLogger(__name__)

LABEL_KNOWS = "KNOWS"
LABEL_FAKING = "FAKING"
SECTION_PATTERN = re.compile(r"(\d+)\s*[:=\-]\s*(KNOWS|FAKING)", re.IGNORECASE)
ASK_PATTERN = re.compile(
    r"section[_\s]+(\d+)[:\s]+(.+?)(?=section[_\s]+\d+|classify|$)",
    re.IGNORECASE | re.DOTALL,
)

REQUIRED_WANDB_METRICS = (
    "total_reward",
    "accuracy",
    "false_accusations",
    "efficiency_score",
    "mean_turns_to_classify",
    "loss",
    "kl_divergence",
)

DEMO_QUESTIONS = [
    "Explain the core mechanism of this concept in one sentence.",
    "Why does this approach fail on edge cases?",
    "What is the causal chain that produces the main effect?",
    "Give a specific numerical or mechanistic example.",
    "How does this interact with gradient flow during training?",
    "What breaks if you remove the key constraint?",
]


def _parse_partition_from_completion(completion: str, n_sections: int = 10) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for match in SECTION_PATTERN.finditer(completion):
        idx = int(match.group(1))
        if 1 <= idx <= n_sections:
            parsed[idx - 1] = match.group(2).upper()
    for idx in range(n_sections):
        parsed.setdefault(idx, LABEL_KNOWS)
    return parsed


def _parse_questions_from_completion(completion: str, n_sections: int = 10) -> List[Tuple[int, str]]:
    questions: List[Tuple[int, str]] = []
    for match in ASK_PATTERN.finditer(completion):
        sid = int(match.group(1))
        question = match.group(2).strip()[:300]
        if 0 <= sid < n_sections and question:
            questions.append((sid, question))
    return questions


def _ensure_finite(metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise RuntimeError(f"Non-finite metric detected: {key}={value}")


def _validate_metric_payload(payload: Dict[str, float]) -> None:
    missing = [key for key in REQUIRED_WANDB_METRICS if key not in payload]
    if missing:
        raise RuntimeError(f"Missing required metrics: {missing}")
    _ensure_finite(payload)


async def _run_episode_async(
    completion: str,
    cfg: TrainingConfig,
) -> Dict[str, float]:
    """Run a real ExaminerEnvironment episode from a model completion.

    Parses ask/classify actions from completion text, runs them against the
    local environment, returns episode metrics.
    """
    from examiner_env.models import ExaminerAction
    from examiner_env.server.examiner_environment import ExaminerEnvironment

    env = ExaminerEnvironment()
    obs = await env.reset()
    n_sections = cfg.env_sections

    questions = _parse_questions_from_completion(completion, n_sections=n_sections)
    if not questions:
        for turn in range(min(3, cfg.max_turns - 1)):
            sid = turn % n_sections
            questions.append((sid, DEMO_QUESTIONS[turn % len(DEMO_QUESTIONS)]))

    turns_used = 0
    for sid, question_text in questions[: cfg.max_turns - 1]:
        action = ExaminerAction(
            action_type="ask",
            section_id=sid,
            question_text=question_text,
        )
        result = await env.step(action)
        turns_used += 1
        if result.done:
            metadata = result.observation.metadata or {}
            return {
                "total_reward": float(result.reward or -0.5),
                "accuracy": float(metadata.get("accuracy", 0.0)),
                "false_accusations": float(metadata.get("false_accusations", 0)),
                "efficiency_score": float(
                    max(cfg.max_turns - turns_used, 0) / cfg.max_turns
                ),
                "mean_turns_to_classify": float(turns_used),
            }

    classification = _parse_partition_from_completion(completion, n_sections=n_sections)
    classify_action = ExaminerAction(
        action_type="classify",
        classification=classification,
    )
    result = await env.step(classify_action)
    turns_used += 1
    metadata = result.observation.metadata or {}

    return {
        "total_reward": float(result.reward or 0.0),
        "accuracy": float(metadata.get("accuracy", 0.0)),
        "false_accusations": float(metadata.get("false_accusations", 0)),
        "efficiency_score": float(
            max(cfg.max_turns - turns_used, 0) / cfg.max_turns
        ),
        "mean_turns_to_classify": float(turns_used),
    }


def _run_episode_sync(completion: str, cfg: TrainingConfig) -> Dict[str, float]:
    """Sync wrapper around async episode runner."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(_run_episode_async(completion, cfg))
    except Exception:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run_episode_async(completion, cfg))
        finally:
            loop.close()


class EnvironmentRewardAdapter:
    """C2-owned adapter: runs real ExaminerEnvironment episodes.

    Falls back to offline scoring if env import fails (e.g. during unit tests
    before C1 env is installed). Set cfg.env_adapter_mode='offline' to force.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._env_available = self._check_env_available()
        if not self._env_available:
            LOGGER.warning(
                "ExaminerEnvironment not available. Using offline scoring. "
                "Install examiner_env package before real training."
            )

    @staticmethod
    def _check_env_available() -> bool:
        try:
            from examiner_env.server.examiner_environment import ExaminerEnvironment  # noqa: F401
            return True
        except Exception:
            return False

    def _offline_score(self, completion: str) -> Dict[str, float]:
        n = self.cfg.env_sections
        truth = {idx: (LABEL_KNOWS if idx % 2 == 0 else LABEL_FAKING) for idx in range(n)}
        predicted = _parse_partition_from_completion(completion, n_sections=n)
        correct = 0
        false_accusations = 0
        for section_id in truth:
            p = predicted.get(section_id, LABEL_KNOWS)
            t = truth[section_id]
            if p == t:
                correct += 1
            elif p == LABEL_FAKING and t == LABEL_KNOWS:
                false_accusations += 1
                correct -= 1
        accuracy_score = correct / max(n, 1)
        efficiency_score = max(self.cfg.max_turns - 6, 0) / self.cfg.max_turns
        total_reward = 0.70 * accuracy_score - 0.50 * false_accusations + 0.20 * efficiency_score
        return {
            "total_reward": float(total_reward),
            "accuracy": float(max((correct + n) / (2 * n), 0.0)),
            "false_accusations": float(false_accusations),
            "efficiency_score": float(efficiency_score),
            "mean_turns_to_classify": 6.0,
        }

    def evaluate_completion(self, completion: str) -> Dict[str, float]:
        use_real = (
            self._env_available
            and self.cfg.env_adapter_mode != "offline"
        )
        if use_real:
            try:
                return _run_episode_sync(completion, self.cfg)
            except Exception as exc:
                LOGGER.warning("Real env episode failed (%s). Falling back to offline.", exc)
        return self._offline_score(completion)


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


def _default_dataset(cfg: TrainingConfig) -> List[Dict[str, str]]:
    prompt = (
        "You are an expert Examiner. You will see section titles of a knowledge base. "
        "Ask diagnostic questions (format: 'Section <id>: <question>') then classify "
        "each section (format: '<id>: KNOWS or FAKING').\n\n"
        f"{cfg.question_prompt}"
    )
    return [{"prompt": prompt}] * cfg.max_steps


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
    reward_fn = build_reward_fn(cfg)
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=train_cfg,
        train_dataset=_default_dataset(cfg),
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

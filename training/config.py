from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class TrainingConfig:
    examiner_model: str = "Qwen/Qwen2.5-7B-Instruct"
    student_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    wandb_project: str = "the-examiner"
    hf_hub_repo: str = "team/the-examiner"
    output_dir: str = "outputs/checkpoints"
    merged_checkpoint_dir: str = "examiner_checkpoint"

    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    learning_rate: float = 5e-6
    max_steps: int = 500
    per_device_batch_size: int = 1
    grad_accumulation_steps: int = 1
    warmup_steps: int = 20
    weight_decay: float = 0.01
    logging_steps: int = 1
    eval_every: int = 50
    save_every: int = 100
    num_generations: int = 4
    kl_penalty: float = 0.01
    seed: int = 42
    use_wandb: bool = True
    strict_metrics: bool = True
    resume_from_checkpoint: str | None = None

    max_turns: int = 20
    smoke_eval_episodes: int = 20
    env_base_url: str = "http://127.0.0.1:8000"
    env_sections: int = 10
    env_adapter_mode: str = "offline"
    question_prompt: str = (
        "Ask one diagnostic question that can separate genuine knowledge from faking."
    )

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.eval_every <= 0 or self.save_every <= 0:
            raise ValueError("eval_every and save_every must be > 0")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be > 0")
        if self.num_generations <= 0:
            raise ValueError("num_generations must be > 0")
        if not (0.0 <= self.kl_penalty <= 1.0):
            raise ValueError("kl_penalty must be in [0, 1]")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be > 0")
        if self.env_sections <= 0:
            raise ValueError("env_sections must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

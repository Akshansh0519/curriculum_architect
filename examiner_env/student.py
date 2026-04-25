from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Literal, Optional


Label = Literal["KNOWS", "FAKING"]


@dataclass
class ScriptedStudent:
    student_model_id: str
    partition: Dict[int, Label]

    def __post_init__(self) -> None:
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe
        # Never auto-download large models during local smoke tests / CI.
        # Opt-in if you actually want HF model inference here.
        if os.getenv("EXAMINER_ENABLE_HF_STUDENT", "").lower() not in {"1", "true", "yes"}:
            self._pipe = None
            return None
        try:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline(
                "text-generation",
                model=self.student_model_id,
                device_map="auto",
                local_files_only=True,
            )
            return self._pipe
        except Exception:
            self._pipe = None
            return None

    def generate_answer(self, section_title: str, section_id: int, question: str) -> str:
        mode: Label = self.partition.get(section_id, "FAKING")

        if mode == "KNOWS":
            system = (
                f"You are an expert who has genuinely mastered {section_title}. "
                "Answer with mechanistic depth. Explain WHY things work, not just WHAT they are. "
                "Give specific examples. Acknowledge uncertainty where it exists. Reference edge cases."
            )
        else:
            system = (
                f"You have only surface-level familiarity with {section_title}. "
                "Your goal is to SOUND confident and knowledgeable. Use correct terminology. "
                "Give definitional answers. NEVER admit uncertainty. "
                "AVOID committing to specific mechanisms or causal chains. "
                "If asked about edge cases, give plausible-sounding but vague answers."
            )

        prompt = f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n"

        pipe = self._get_pipe()
        if pipe is None:
            # Fallback: no model available locally.
            if mode == "KNOWS":
                return f"{section_title}: Mechanistically, {question} depends on the underlying optimization/gradient dynamics and assumptions."
            return f"{section_title} is a widely used concept in ML; it helps improve performance and stability in practice."

        try:
            out = pipe(
                prompt,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.8 if mode == "FAKING" else 0.6,
                top_p=0.9,
                repetition_penalty=1.05,
                return_full_text=False,
            )
            if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                return str(out[0]["generated_text"]).strip()
        except Exception:
            pass

        return "I can’t generate an answer right now."


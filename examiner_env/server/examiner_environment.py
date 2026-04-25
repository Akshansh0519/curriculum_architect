# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The Examiner — OpenEnv environment core.
"""

from __future__ import annotations

from uuid import uuid4
import random
from typing import Dict, Literal, Optional, Tuple

from openenv.core import Environment
from openenv.core.client_types import StepResult

try:
    from openenv.core.env_server.types import State
except Exception:  # pragma: no cover
    from openenv.core.env_server.types import State  # type: ignore

try:
    from ..knowledge_base import KnowledgeBase
    from ..models import ExaminerAction, ExaminerObservation, ExaminerState
    from ..reward import compute_reward
    from ..student import ScriptedStudent
except ImportError:
    from knowledge_base import KnowledgeBase  # type: ignore
    from models import ExaminerAction, ExaminerObservation, ExaminerState  # type: ignore
    from reward import compute_reward  # type: ignore
    from student import ScriptedStudent  # type: ignore


Label = Literal["KNOWS", "FAKING"]


class ExaminerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._kb: Optional[KnowledgeBase] = None
        self._partition: Dict[int, Label] = {}
        self._genuine_baselines: Dict[int, str] = {}
        self._history: list = []
        self._turn_counter: int = 0
        self._max_turns: int = 20
        self._rng = random.Random()
        self._student: Optional[ScriptedStudent] = None

        self._state = ExaminerState(episode_id=str(uuid4()), step_count=0, max_turns=self._max_turns)

    def _do_reset(self, seed: Optional[int], episode_id: Optional[str], **kwargs) -> ExaminerObservation:
        self._rng = random.Random(seed)
        self._kb = KnowledgeBase(domain=kwargs.get("kb_domain", "ml_theory"))
        self._history = []
        self._turn_counter = 0

        k = self._rng.choice([3, 4, 5])
        knows = set(self._rng.sample(list(range(10)), k=k))
        self._partition = {i: ("KNOWS" if i in knows else "FAKING") for i in range(10)}

        student_model_id = kwargs.get("student_model_id", "meta-llama/Llama-3.2-3B-Instruct")
        self._student = ScriptedStudent(student_model_id=student_model_id, partition=self._partition)

        # Baselines used only for reward diagnostic-quality proxy; keep fully programmatic.
        self._genuine_baselines = {}
        for sid, label in self._partition.items():
            if label != "FAKING":
                continue
            section = self._kb.get_section(sid)
            self._genuine_baselines[sid] = " ".join(section.key_concepts[:3])

        self._state = ExaminerState(episode_id=episode_id or str(uuid4()), step_count=0, max_turns=self._max_turns)

        return ExaminerObservation(
            section_titles=self._kb.section_titles(),
            question_history=[],
            turn_counter=0,
            remaining_turns=self._max_turns,
            belief_scratchpad="",
            done=False,
            reward=0.0,
            metadata={"partition_k": k},
        )

    def _do_step(self, action: ExaminerAction) -> Tuple[ExaminerObservation, Optional[float], bool]:
        if self._kb is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._state.step_count += 1

        if action.action_type == "ask":
            if action.section_id is None or action.question_text is None:
                raise ValueError("ask requires section_id and question_text")

            sid = int(action.section_id)
            section = self._kb.get_section(sid)
            answer = (
                self._student.generate_answer(section.title, sid, action.question_text) if self._student else ""
            )

            self._history.append({"section_id": str(sid), "question": action.question_text, "answer": answer})
            self._turn_counter += 1

            remaining = max(0, self._max_turns - self._turn_counter)
            forced_done = self._turn_counter >= self._max_turns

            obs = ExaminerObservation(
                section_titles=self._kb.section_titles(),
                question_history=self._history,
                turn_counter=self._turn_counter,
                remaining_turns=remaining,
                belief_scratchpad="",
                done=forced_done,
                reward=(-0.5 if forced_done else 0.0),
                metadata={"forced_terminate": forced_done},
            )
            return obs, float(obs.reward or 0.0), bool(obs.done)

        if action.action_type == "classify":
            if not action.classification:
                raise ValueError("classify requires classification dict")

            predicted = {int(k): v for k, v in action.classification.items()}
            turns_used = self._turn_counter
            r = compute_reward(
                predicted=predicted,
                true=self._partition,
                turns_used=turns_used,
                max_turns=self._max_turns,
                question_history=self._history,
                genuine_baselines=self._genuine_baselines,
            )

            # Simple metrics mirror reward components.
            false_acc = sum(1 for i in range(10) if predicted.get(i) == "FAKING" and self._partition.get(i) == "KNOWS")
            correct = 0
            for i in range(10):
                p = predicted.get(i)
                t = self._partition.get(i)
                if p == t and p is not None:
                    correct += 1
            acc01 = correct / 10.0

            obs = ExaminerObservation(
                section_titles=self._kb.section_titles(),
                question_history=self._history,
                turn_counter=self._turn_counter,
                remaining_turns=max(0, self._max_turns - self._turn_counter),
                belief_scratchpad="",
                done=True,
                reward=float(r),
                metadata={
                    "accuracy": acc01,
                    "false_accusations": false_acc,
                    "turns_used": turns_used,
                },
            )
            return obs, float(obs.reward or 0.0), bool(obs.done)

        raise ValueError(f"Unknown action_type: {action.action_type}")

    # ---------- Async-first API (validator gate expects this) ----------
    async def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ExaminerObservation:
        return self._do_reset(seed=seed, episode_id=episode_id, **kwargs)

    async def step(self, action: ExaminerAction, timeout_s: Optional[float] = None, **kwargs) -> StepResult[ExaminerObservation]:  # type: ignore[override]
        obs, reward, done = self._do_step(action)
        return StepResult(observation=obs, reward=reward, done=done)

    async def state(self) -> ExaminerState:
        return self._state

    # ---------- OpenEnv HTTP server compatibility ----------
    async def reset_async(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ExaminerObservation:  # type: ignore[override]
        # HTTPEnvServer checks override of reset_async/step_async; it expects Observation, not StepResult.
        return self._do_reset(seed=seed, episode_id=episode_id, **kwargs)

    async def step_async(self, action: ExaminerAction, timeout_s: Optional[float] = None, **kwargs) -> ExaminerObservation:  # type: ignore[override]
        obs, _reward, _done = self._do_step(action)
        return obs

    def render(self) -> str:
        if self._kb is None:
            return "<uninitialized>"
        lines = ["The Examiner — Episode Transcript", ""]
        for idx, qa in enumerate(self._history, start=1):
            lines.append(f"Q{idx} (section {qa.get('section_id')}): {qa.get('question')}")
            lines.append(f"A{idx}: {qa.get('answer')}")
            lines.append("")
        return "\n".join(lines).strip()

    def get_metrics(self) -> dict:
        # Available after classify (or forced termination).
        if not self._partition:
            return {}
        turns_used = self._turn_counter
        eff = ((self._max_turns - turns_used) / self._max_turns) if self._max_turns else 0.0
        return {
            "turns_used": turns_used,
            "efficiency": eff,
        }

    @property
    def state(self) -> State:
        return self._state

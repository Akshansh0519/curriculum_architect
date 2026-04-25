"""Examiner Env client — C2 owned.

Wraps the OpenEnv EnvClient to use real ExaminerAction / ExaminerObservation
types as shipped in C1's examiner_env/models.py.
"""

from __future__ import annotations

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ExaminerAction, ExaminerObservation, ExaminerState


class ExaminerEnv(EnvClient[ExaminerAction, ExaminerObservation, State]):
    """WebSocket client for the live ExaminerEnvironment server.

    Usage::

        async with ExaminerEnv(base_url="http://localhost:8000") as client:
            obs = await client.reset()
            result = await client.step(
                ExaminerAction(action_type="ask", section_id=0,
                               question_text="Explain gradient descent.")
            )
            result = await client.step(
                ExaminerAction(action_type="classify",
                               classification={i: "KNOWS" for i in range(10)})
            )
    """

    def _step_payload(self, action: ExaminerAction) -> Dict:
        payload: Dict = {"action_type": action.action_type}
        if action.action_type == "ask":
            payload["section_id"] = action.section_id
            payload["question_text"] = action.question_text
        elif action.action_type == "classify":
            payload["classification"] = {
                str(k): v for k, v in (action.classification or {}).items()
            }
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[ExaminerObservation]:
        obs_data = payload.get("observation", {})
        observation = ExaminerObservation(
            section_titles=obs_data.get("section_titles", []),
            question_history=obs_data.get("question_history", []),
            turn_counter=obs_data.get("turn_counter", 0),
            remaining_turns=obs_data.get("remaining_turns", 20),
            belief_scratchpad=obs_data.get("belief_scratchpad", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ExaminerState:
        return ExaminerState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            max_turns=payload.get("max_turns", 20),
        )


async def run_local_episode(
    difficulty: str = "medium",
    max_questions: Optional[int] = 5,
) -> Dict:
    """Run a full episode against a local ExaminerEnvironment instance.

    Returns a dict with transcript, classification result, ground truth, and reward.
    Used by hf_space/app.py Live Demo tab.
    """
    from examiner_env.server.examiner_environment import ExaminerEnvironment

    difficulty_turns = {"easy": 3, "medium": 5, "hard": 8}
    n_questions = difficulty_turns.get(difficulty, 5)
    if max_questions is not None:
        n_questions = min(n_questions, max_questions)

    env = ExaminerEnvironment()
    obs = await env.reset()

    transcript: list = []
    section_titles = obs.section_titles

    demo_questions = [
        "Explain the core mechanism of this topic in one sentence.",
        "Why does this approach fail on edge cases?",
        "What is the causal chain that produces the main effect?",
        "Give a specific numerical or mechanistic example.",
        "How does this interact with gradient flow during training?",
        "What breaks if you remove the key constraint?",
        "Distinguish between two common misconceptions.",
        "When does the standard assumption fail?",
    ]

    for turn in range(n_questions):
        section_id = turn % len(section_titles)
        question = demo_questions[turn % len(demo_questions)]
        action = ExaminerAction(
            action_type="ask",
            section_id=section_id,
            question_text=question,
        )
        result = await env.step(action)
        history = result.observation.question_history
        last = history[-1] if history else {}
        transcript.append({
            "speaker": "Examiner",
            "text": f"[Section {section_id}: {section_titles[section_id]}] {question}",
        })
        transcript.append({
            "speaker": "Student",
            "text": last.get("answer", "(no answer)"),
        })
        if result.done:
            break

    classification = {i: "KNOWS" for i in range(len(section_titles))}
    classify_action = ExaminerAction(
        action_type="classify",
        classification=classification,
    )
    final_result = await env.step(classify_action)
    final_obs = final_result.observation
    metadata = final_obs.metadata or {}

    return {
        "transcript": transcript,
        "reward": float(final_result.reward or 0.0),
        "accuracy": metadata.get("accuracy", 0.0),
        "false_accusations": metadata.get("false_accusations", 0),
        "turns_used": metadata.get("turns_used", n_questions),
        "done": True,
    }

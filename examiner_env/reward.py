from __future__ import annotations

import re
from typing import Dict, Iterable, List, Literal, Optional, Set


Label = Literal["KNOWS", "FAKING"]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def compute_answer_divergence(answer_a: str, answer_b: str) -> float:
    """
    Proxy divergence: 1 - (shared_tokens / total_unique_tokens).
    Higher => more different => better diagnostic question.
    """
    a = _tokens(answer_a)
    b = _tokens(answer_b)
    if not a and not b:
        return 0.0
    shared = len(a & b)
    total = len(a | b)
    return 1.0 - (shared / total)


def compute_reward(
    predicted: Dict[int, Label],
    true: Dict[int, Label],
    turns_used: int,
    max_turns: int,
    question_history: List[dict],
    genuine_baselines: Dict[int, str],
) -> float:
    # Component 1: classification accuracy (weight: 0.70)
    correct = 0
    false_accusations = 0
    for i in range(10):
        p = predicted.get(i)
        t = true.get(i)
        if p == t and p is not None:
            correct += 1
        elif p == "FAKING" and t == "KNOWS":
            false_accusations += 1
            correct -= 1  # penalize in accuracy too
    accuracy_score = correct / 10.0  # [-1, 1] (by design)

    # Component 2: false accusation penalty (weight: -0.50 per error)
    fa_penalty = false_accusations * 0.5

    # Component 3: efficiency bonus (weight: 0.20)
    turns_used = max(0, min(int(turns_used), int(max_turns)))
    efficiency_bonus = ((max_turns - turns_used) / max_turns) * 0.20 if max_turns else 0.0

    # Component 4: diagnostic quality bonus (weight: 0.10)
    # Only for sections that are truly FAKING and were asked about.
    per_section_last_answer: Dict[int, str] = {}
    for item in question_history:
        try:
            sid = int(item.get("section_id"))
        except Exception:
            continue
        ans = item.get("answer")
        if isinstance(ans, str):
            per_section_last_answer[sid] = ans

    divergences: List[float] = []
    for sid, t in true.items():
        if t != "FAKING":
            continue
        if sid not in per_section_last_answer:
            continue
        baseline = genuine_baselines.get(sid, "")
        divergences.append(compute_answer_divergence(per_section_last_answer[sid], baseline))

    diagnostic_score = 0.0
    if divergences:
        diagnostic_score = (sum(divergences) / len(divergences)) * 0.10

    # Final reward
    return (0.70 * accuracy_score) - fa_penalty + efficiency_bonus + diagnostic_score


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for The Examiner environment.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ExaminerAction(Action):
    action_type: Literal["ask", "classify"] = Field(..., description="Ask a question or submit a full classification.")
    section_id: Optional[int] = Field(
        default=None, description="Target section_id for an ask action (0..9). Required when action_type='ask'."
    )
    question_text: Optional[str] = Field(
        default=None, description="Question text for an ask action. Required when action_type='ask'."
    )
    classification: Optional[Dict[int, Literal["KNOWS", "FAKING"]]] = Field(
        default=None,
        description="Mapping section_id -> KNOWS|FAKING. Required when action_type='classify'.",
    )


class ExaminerObservation(Observation):
    section_titles: List[str] = Field(..., description="Titles of the 10 KB sections (visible to Examiner).")
    question_history: List[Dict[str, str]] = Field(
        default_factory=list, description="List of {section_id, question, answer} dicts."
    )
    turn_counter: int = Field(default=0, description="Number of questions asked so far.")
    remaining_turns: int = Field(default=20, description="Remaining turns before forced termination.")
    belief_scratchpad: str = Field(default="", description="Examiner scratchpad (working memory).")


class ExaminerState(State):
    max_turns: int = Field(default=20, description="Max turns for this episode.")

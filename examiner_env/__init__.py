# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Examiner environment package.

Keep imports lightweight: do not import network clients at import-time.
"""

from .models import ExaminerAction, ExaminerObservation, ExaminerState

__all__ = ["ExaminerAction", "ExaminerObservation", "ExaminerState", "ExaminerEnv"]


def __getattr__(name: str):
    if name == "ExaminerEnv":
        # Lazy import (client pulls in websocket/http dependencies).
        from .client import ExaminerEnv  # type: ignore

        return ExaminerEnv
    raise AttributeError(name)

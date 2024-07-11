from __future__ import annotations

from typing import Any
from instructor.usage import UnifiedUsage


class IncompleteOutputException(Exception):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(
        self,
        *args: list[Any],
        last_completion: Any | None = None,
        message: str = "The output is incomplete due to a max_tokens length limit.",
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        super().__init__(message, *args, **kwargs)


class InstructorRetryException(Exception):
    def __init__(
        self,
        *args: list[Any],
        last_completion: Any | None = None,
        messages: list[Any] | None = None,
        n_attempts: int,
        total_usage: UnifiedUsage,
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        self.messages = messages
        self.n_attempts = n_attempts
        self.total_usage = total_usage
        super().__init__(*args, **kwargs)

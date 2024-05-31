# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload

import google.generativeai as genai

import instructor


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert (
        mode == instructor.Mode.GEMINI_JSON
    ), "Mode must be instructor.Mode.GEMINI_JSON"

    assert isinstance(
        client,
        (genai.GenerativeModel),
    ), "Client must be an instance of genai.generativemodel"

    if use_async:
        create = client.generate_content_async
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.GEMINI,
            mode=mode,
            **kwargs,
        )
    else:
        create = client.generate_content
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.GEMINI,
            mode=mode,
            **kwargs,
        )

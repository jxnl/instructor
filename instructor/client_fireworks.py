from __future__ import annotations

from typing import Any, Literal, overload

import instructor
from instructor.client import AsyncInstructor, Instructor


from fireworks.client import Fireworks, AsyncFireworks
from fireworks.client.api_client_v2 import Fireworks as FireworksType, AsyncFireworks as AsyncFireworksType

@overload
def from_fireworks(
    client: FireworksType,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_fireworks(
    client: AsyncFireworksType,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_fireworks(
    client: FireworksType | AsyncFireworksType,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    assert (
        mode
        in {
            instructor.Mode.FIREWORKS_TOOLS,
            instructor.Mode.FIREWORKS_JSON,
        }
    ), "Mode must be one of {instructor.Mode.FIREWORKS_TOOLS, instructor.Mode.FIREWORKS_JSON}"

    assert isinstance(
        client, (FireworksType, AsyncFireworksType)
    ), "Client must be an instance of Fireworks or AsyncFireworks"

    if use_async:
        create = client.chat.completions.create
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.FIREWORKS,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.FIREWORKS,
        mode=mode,
        **kwargs,
    )

from __future__ import annotations

import cohere
import instructor
from typing import (
    TypeVar,
    overload,
)
from typing import Any
from typing_extensions import ParamSpec
from pydantic import BaseModel


T_Model = TypeVar("T_Model", bound=BaseModel)
T_ParamSpec = ParamSpec("T_ParamSpec")


@overload
def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_JSON_SCHEMA,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_cohere(
    client: cohere.Client | cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
):
    assert mode in {
        instructor.Mode.COHERE_TOOLS,
        instructor.Mode.COHERE_JSON_SCHEMA,
    }, "Mode be one of {COHERE_TOOLS, COHERE_JSON_SCHEMA}"

    assert isinstance(
        client, (cohere.Client, cohere.AsyncClient)
    ), "Client must be an instance of cohere.Cohere or cohere.AsyncCohere"

    if isinstance(client, cohere.Client):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat, mode=mode),
            provider=instructor.Provider.COHERE,
            mode=mode,
            **kwargs,
        )

    if isinstance(client, cohere.AsyncClient):
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat, mode=mode),
            provider=instructor.Provider.COHERE,
            mode=mode,
            **kwargs,
        )

import logging
from typing import (
    Callable,
    Union,
    overload,
)

from openai import AsyncOpenAI, OpenAI
from instructor.client import InstructorAsyncOpenAI, InstructorOpenAI

from .function_calls import Mode

logger = logging.getLogger("instructor")


@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.FUNCTIONS,
) -> InstructorOpenAI:
    ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.FUNCTIONS,
) -> InstructorAsyncOpenAI:
    ...


@overload
def patch(
    create: Callable,
    mode: Mode = Mode.FUNCTIONS,
):
    ...


def patch(
    client: Union[OpenAI, AsyncOpenAI] = None,
    create: Callable = None,
    mode: Mode = Mode.FUNCTIONS,
) -> Union[InstructorOpenAI, InstructorAsyncOpenAI]:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    if client is not None:
        logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

        if isinstance(client, AsyncOpenAI):
            return InstructorAsyncOpenAI(client, mode=mode)
        else:
            return InstructorOpenAI(client, mode=mode)

    if create is not None:
        raise NotImplementedError("Patching a function is not yet supported in v1.0.0")


def apatch(client: AsyncOpenAI, mode: Mode = Mode.FUNCTIONS):
    """
    No longer necessary, use `patch` instead.

    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """
    import warnings

    warnings.warn(
        "apatch is deprecated, use patch instead", DeprecationWarning, stacklevel=2
    )
    return patch(client, mode=mode)

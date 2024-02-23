import logging
from tenacity import Retrying, AsyncRetrying, stop_after_attempt, RetryError
from json import JSONDecodeError
from typing import (
    Callable,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
)

from openai.types.chat import (
    ChatCompletion,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError


from instructor.function_calls import Mode
from instructor.response import process_response, process_response_async
from instructor.utils import dump_message

logger = logging.getLogger("instructor")

T = TypeVar("T")
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")


async def retry_async(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: Type[T],
    validation_context,
    args,
    kwargs,
    max_retries: int | AsyncRetrying = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.TOOLS,
) -> T:
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    # If max_retries is int, then create a AsyncRetrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (AsyncRetrying, Retrying)):
        raise ValueError(
            "max_retries must be an `int` or a `tenacity.AsyncRetrying` object"
        )

    try:
        async for attempt in max_retries:
            logger.debug(f"Retrying, attempt: {attempt}")
            with attempt:
                try:
                    response: ChatCompletion = await func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    if (
                        isinstance(response, ChatCompletion)
                        and response.usage is not None
                    ):
                        total_usage.completion_tokens += (
                            response.usage.completion_tokens or 0
                        )
                        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                        total_usage.total_tokens += response.usage.total_tokens or 0
                        response.usage = total_usage  # Replace each response usage with the total usage
                    return await process_response_async(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}")
                    kwargs["messages"].append(dump_message(response.choices[0].message))  # type: ignore
                    if mode == Mode.TOOLS:
                        kwargs["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": response.choices[0]
                                .message.tool_calls[0]
                                .id,
                                "name": response.choices[0]
                                .message.tool_calls[0]
                                .function.name,
                                "content": "Exceptions found\n{e}\nRecall the function correctly.",
                            }
                        )

                    kwargs["messages"].append(
                        {
                            "role": "user",
                            "content": f"Recall the function correctly, fix the errors, exceptions found\n{e}",
                        }
                    )
                    if mode == Mode.MD_JSON:
                        kwargs["messages"].append(
                            {
                                "role": "assistant",
                                "content": "```json",
                            },
                        )
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e


def retry_sync(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: Type[T],
    validation_context: dict,
    args,
    kwargs,
    max_retries: int | Retrying = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.TOOLS,
):
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    # If max_retries is int, then create a Retrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries: Retrying = Retrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (Retrying, AsyncRetrying)):
        raise ValueError("max_retries must be an int or a `tenacity.Retrying` object")

    try:
        for attempt in max_retries:
            with attempt:
                try:
                    response = func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    if (
                        isinstance(response, ChatCompletion)
                        and response.usage is not None
                    ):
                        total_usage.completion_tokens += (
                            response.usage.completion_tokens or 0
                        )
                        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                        total_usage.total_tokens += response.usage.total_tokens or 0
                        response.usage = total_usage  # Replace each response usage with the total usage
                    return process_response(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}")
                    kwargs["messages"].append(dump_message(response.choices[0].message))
                    # ! How do we handle this for parallel tools in the future?
                    if mode == Mode.TOOLS:
                        kwargs["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": response.choices[0]
                                .message.tool_calls[0]
                                .id,
                                "name": response.choices[0]
                                .message.tool_calls[0]
                                .function.name,
                                "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                            }
                        )
                    else:
                        kwargs["messages"].append(
                            {
                                "role": "user",
                                "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                            }
                        )
                    if mode == Mode.MD_JSON:
                        kwargs["messages"].append(
                            {
                                "role": "assistant",
                                "content": "```json",
                            },
                        )
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e

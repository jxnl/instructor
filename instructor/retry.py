# type: ignore[all]
import logging

from openai.types.chat import ChatCompletion
from instructor.mode import Mode
from instructor.process_response import process_response, process_response_async
from instructor.utils import dump_message, update_total_usage

from openai.types.completion_usage import CompletionUsage
from pydantic import ValidationError
from tenacity import AsyncRetrying, RetryError, Retrying, stop_after_attempt


from json import JSONDecodeError
from pydantic import BaseModel
from typing import Callable, Optional, Type, TypeVar, ParamSpec

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def reask_messages(response: ChatCompletion, mode: Mode, exception: Exception):
    if mode == Mode.ANTHROPIC_TOOLS:
        # TODO: we need to include the original response
        yield {
            "role": "user",
            "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
        }
        return

    yield dump_message(response.choices[0].message)
    # TODO: Give users more control on configuration
    if mode == Mode.TOOLS:
        for tool_call in response.choices[0].message.tool_calls:  # type: ignore
            yield {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
            }
    elif mode == Mode.MD_JSON:
        yield {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    else:
        yield {
            "role": "user",
            "content": f"Recall the function correctly, fix the errors, exceptions found\n{exception}",
        }


def retry_sync(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: Type[T_Model],
    validation_context: dict,
    args,
    kwargs,
    max_retries: int | Retrying = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model:
    total_usage = CompletionUsage(
        messages=kwargs.get("messages", []),
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
    )

    # If max_retries is int, then create a Retrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = Retrying(
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
                    response = update_total_usage(
                        response, total_usage, kwargs["messages"]
                    )
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
                    kwargs["messages"].extend(reask_messages(response, mode, e))
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e


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
    total_usage = CompletionUsage(
        messages=kwargs.get("messages", []),
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
    )

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
                    response: ChatCompletion = await func(*args, **kwargs)  # type: ignore
                    stream = kwargs.get("stream", False)
                    response = update_total_usage(
                        response, total_usage, kwargs["messages"]
                    )
                    return await process_response_async(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )  # type: ignore[all]
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}", e)
                    kwargs["messages"].extend(reask_messages(response, mode, e))
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e

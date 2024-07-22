# type: ignore[all]
from __future__ import annotations

import logging

from openai.types.chat import ChatCompletion
from instructor.mode import Mode
from instructor.process_response import process_response, process_response_async
from instructor.utils import (
    dump_message,
    update_total_usage,
    merge_consecutive_messages,
)
from instructor.exceptions import InstructorRetryException

from openai.types.completion_usage import CompletionUsage
from pydantic import ValidationError
from tenacity import AsyncRetrying, RetryError, Retrying, stop_after_attempt
from instructor.validators import AsyncValidationError

from json import JSONDecodeError
from pydantic import BaseModel
from typing import Callable, TypeVar, Any
from typing_extensions import ParamSpec

import logfire_api as logfire

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def reask_messages(response: ChatCompletion, mode: Mode, exception: Exception):
    if mode == Mode.ANTHROPIC_TOOLS:
        # The original response
        assistant_content = []
        tool_use_id = None
        for content in response.content:
            assistant_content.append(content.model_dump())
            # Assuming exception from single tool invocation
            if (
                content.type == "tool_use"
                and isinstance(exception, ValidationError)
                and content.name == exception.title
            ):
                tool_use_id = content.id

        yield {
            "role": "assistant",
            "content": assistant_content,
        }
        if tool_use_id is not None:
            yield {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
                        "is_error": True,
                    }
                ],
            }
        else:
            yield {
                "role": "user",
                "content": f"Validation Error due to no tool invocation:\n{exception}\nRecall the function correctly, fix the errors",
            }
        return
    if mode == Mode.ANTHROPIC_JSON:
        from anthropic.types import Message

        assert isinstance(response, Message)
        yield {
            "role": "user",
            "content": f"""Validation Errors found:\n{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{response.content[0].text}""",
        }
        return
    if mode == Mode.COHERE_TOOLS:
        yield {
            "role": "user",
            "message": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
        }
        return
    if mode == Mode.GEMINI_JSON:
        yield {
            "role": "user",
            "parts": [
                f"Correct the following JSON response, based on the errors given below:\n\nJSON:\n{response.text}\n\nExceptions:\n{exception}"
            ],
        }
        return
    if mode == Mode.VERTEXAI_TOOLS:
        from .client_vertexai import vertexai_function_response_parser

        yield response.candidates[0].content
        yield vertexai_function_response_parser(response, exception)
        return
    if mode == Mode.VERTEXAI_JSON:
        from .client_vertexai import vertexai_message_parser

        yield response.candidates[0].content
        yield vertexai_message_parser(
            {
                "role": "user",
                "content": f"Validation Errors found:\n{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{response.text}",
            }
        )
        return

    yield dump_message(response.choices[0].message)
    # TODO: Give users more control on configuration
    if mode == Mode.TOOLS:
        for tool_call in response.choices[0].message.tool_calls:
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
    response_model: type[T_Model],
    validation_context: dict,
    args,
    kwargs,
    max_retries: int | Retrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model:
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
        from anthropic.types import Usage as AnthropicUsage

        total_usage = AnthropicUsage(input_tokens=0, output_tokens=0)

    # If max_retries is int, then create a Retrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = Retrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (Retrying, AsyncRetrying)):
        raise ValueError("max_retries must be an int or a `tenacity.Retrying` object")

    with logfire.span("Retrying", max_retries=max_retries, mode=mode):
        response = None
        for attempt in max_retries:
            logfire.info(f"Attempt {attempt.retry_state.attempt_number}")
            with attempt:
                try:
                    response = func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    response = update_total_usage(response, total_usage)
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
                    logfire.info(f"Error Response: {response}")
                    if mode in {
                        Mode.GEMINI_JSON,
                        Mode.VERTEXAI_TOOLS,
                        Mode.VERTEXAI_JSON,
                    }:
                        kwargs["contents"].extend(reask_messages(response, mode, e))
                    elif mode in {Mode.COHERE_TOOLS}:
                        kwargs["chat_history"].extend(reask_messages(response, mode, e))
                    else:
                        kwargs["messages"].extend(reask_messages(response, mode, e))
                    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
                        kwargs["messages"] = merge_consecutive_messages(
                            kwargs["messages"]
                        )
                    raise e
                except RetryError as e:
                    logfire.error(f"Retry Error: {e}")
                    raise InstructorRetryException(
                        e,
                        last_completion=response,
                        n_attempts=attempt.retry_state.attempt_number,
                        messages=kwargs.get("messages", kwargs.get("contents")),
                        total_usage=total_usage,
                    ) from e


async def retry_async(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: type[T] | None,
    validation_context: dict[str, Any] | None,
    args: Any,
    kwargs: Any,
    max_retries: int | AsyncRetrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T:
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
        from anthropic.types import Usage as AnthropicUsage

        total_usage = AnthropicUsage(input_tokens=0, output_tokens=0)

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

    with logfire.span(
        "Retrying",
        response_model=response_model.model_json_schema() if response_model else None,
        max_retries=max_retries,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    ):
        try:
            response = None
            async for attempt in max_retries:
                logger.debug(f"Retrying, attempt: {attempt}")
                with attempt:
                    try:
                        response: ChatCompletion = await func(*args, **kwargs)
                        stream = kwargs.get("stream", False)
                        response = update_total_usage(response, total_usage)
                        return await process_response_async(
                            response,
                            response_model=response_model,
                            stream=stream,
                            validation_context=validation_context,
                            strict=strict,
                            mode=mode,
                        )
                    except (
                        ValidationError,
                        JSONDecodeError,
                        AsyncValidationError,
                    ) as e:
                        logfire.info(f"Error response: {response}", exception=e)
                        kwargs["messages"].extend(reask_messages(response, mode, e))
                        if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
                            kwargs["messages"] = merge_consecutive_messages(
                                kwargs["messages"]
                            )
                        raise e
        except RetryError as e:
            logfire.info("Failed after retries", exception=e)
            raise InstructorRetryException(
                e,
                last_completion=response,
                n_attempts=attempt.retry_state.attempt_number,
                messages=kwargs["messages"],
                total_usage=total_usage,
            ) from e

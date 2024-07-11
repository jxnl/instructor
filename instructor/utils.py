from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator, Generator, Iterable
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)
import os

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

logger = logging.getLogger("instructor")
R_co = TypeVar("R_co", covariant=True)

from enum import Enum


class Provider(Enum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    DATABRICKS = "databricks"
    UNKNOWN = "unknown"


def get_provider(base_url: str) -> Provider:
    if "anyscale" in str(base_url):
        return Provider.ANYSCALE
    elif "together" in str(base_url):
        return Provider.TOGETHER
    elif "anthropic" in str(base_url):
        return Provider.ANTHROPIC
    elif "groq" in str(base_url):
        return Provider.GROQ
    elif "openai" in str(base_url):
        return Provider.OPENAI
    elif "mistral" in str(base_url):
        return Provider.MISTRAL
    elif "cohere" in str(base_url):
        return Provider.COHERE
    elif "gemini" in str(base_url):
        return Provider.GEMINI
    elif "databricks" in str(base_url):
        return Provider.DATABRICKS
    elif "vertexai" in str(base_url):
        return Provider.VERTEXAI
    return Provider.UNKNOWN


def extract_json_from_codeblock(content: str) -> str:
    first_paren = content.find("{")
    last_paren = content.rfind("}")
    return content[first_paren : last_paren + 1]


def extract_json_from_stream(chunks: Iterable[str]) -> Generator[str, None, None]:
    capturing = False
    brace_count = 0
    for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    capturing = False
    brace_count = 0
    async for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """
    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
    if (
        hasattr(message, "function_call")
        and message.function_call is not None
        and ret["content"]
    ):
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def is_async(func: Callable[..., Any]) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__  # type: ignore - dynamic
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine


def merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # merge all consecutive user messages into a single message
    new_messages: list[dict[str, Any]] = []
    # Detect whether all messages have a flat content (i.e. all string)
    # Some providers require content to be a string, so we need to check that and behave accordingly
    flat_string = all(isinstance(m["content"], str) for m in messages)
    for message in messages:
        new_content = message["content"]
        if not flat_string and isinstance(new_content, str):
            # If content is not flat, transform it into a list of text
            new_content = [{"type": "text", "text": new_content}]

        if len(new_messages) > 0 and message["role"] == new_messages[-1]["role"]:
            if flat_string:
                # New content is a string
                new_messages[-1]["content"] += f"\n\n{new_content}"
            else:
                # New content is a list
                new_messages[-1]["content"].extend(new_content)
        else:
            new_messages.append(
                {
                    "role": message["role"],
                    "content": new_content,
                }
            )

    return new_messages


class classproperty(Generic[R_co]):
    """Descriptor for class-level properties.

    Examples:
        >>> from instructor.utils import classproperty

        >>> class MyClass:
        ...     @classproperty
        ...     def my_property(cls):
        ...         return cls

        >>> assert MyClass.my_property
    """

    def __init__(self, method: Callable[[Any], R_co]) -> None:
        self.cproperty = method

    def __get__(self, instance: object, cls: type[Any]) -> R_co:
        return self.cproperty(cls)


def transform_to_gemini_prompt(
    messages_chatgpt: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    messages_gemini: list[dict[str, Any]] = []
    system_prompt = ""
    for message in messages_chatgpt:
        if message["role"] == "system":
            system_prompt = message["content"]
        elif message["role"] == "user":
            messages_gemini.append(
                {"role": "user", "parts": [message.get("content", "")]}
            )
        elif message["role"] == "assistant":
            messages_gemini.append(
                {"role": "model", "parts": [message.get("content", "")]}
            )
    if system_prompt:
        messages_gemini[0]["parts"].insert(0, f"*{system_prompt}*")

    return messages_gemini


def disable_pydantic_error_url():
    os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"

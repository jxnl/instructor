import inspect
import json
from typing import Callable, Generator, Iterable, AsyncGenerator, TypeVar

from pydantic import BaseModel

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

T_Model = TypeVar("T_Model", bound=BaseModel)


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


def update_total_usage(
    response: T_Model, total_usage, messages
) -> T_Model | ChatCompletion:
    if isinstance(response, ChatCompletion) and response.usage is not None:
        total_usage.messages = messages
        total_usage.completion_tokens += response.usage.completion_tokens or 0
        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
        total_usage.total_tokens += response.usage.total_tokens or 0
        response.usage = total_usage  # Replace each response usage with the total usage
    return response


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


def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine

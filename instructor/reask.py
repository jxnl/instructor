from __future__ import annotations

import logging
from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.utils import dump_message
from pydantic import BaseModel, ValidationError
from typing_extensions import ParamSpec

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def reask_anthropic_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    from anthropic.types import Message
    from anthropic.types.beta.prompt_caching import PromptCachingBetaMessage

    assert isinstance(response, Message) or isinstance(response, PromptCachingBetaMessage), "Response must be a Anthropic Message"

    assistant_content = []
    tool_use_id = None
    for content in response.content:
        assistant_content.append(content.model_dump())  # type: ignore
        if (
            content.type == "tool_use"
            and isinstance(exception, ValidationError)
            and content.name == exception.title
        ):
            tool_use_id = content.id

    reask_msgs = [{"role": "assistant", "content": assistant_content}]  # type: ignore
    if tool_use_id is not None:
        reask_msgs.append(  # type: ignore
            {
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
        )
    else:
        reask_msgs.append(  # type: ignore
            {
                "role": "user",
                "content": f"Validation Error due to no tool invocation:\n{exception}\nRecall the function correctly, fix the errors",
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_anthropic_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    from anthropic.types import Message
    from anthropic.types.beta.prompt_caching import PromptCachingBetaMessage

    assert isinstance(response, Message) or isinstance(response, PromptCachingBetaMessage), "Response must be a Anthropic Message"

    reask_msg = {
        "role": "user",
        "content": f"""Validation Errors found:\n{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{response.content[0].text}""",  # type: ignore
    }
    kwargs["messages"].append(reask_msg)
    return kwargs


def reask_cohere_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Cohere
    exception: Exception,
):
    chat_history = kwargs.get("chat_history", [])
    chat_history.append({"role": "user", "message": kwargs.get("message")})
    kwargs["chat_history"] = chat_history

    kwargs["message"] = (
        f"Correct the following JSON response, based on the errors given below:\n\n"
        f"JSON:\n{response.text}\n\nExceptions:\n{exception}"
    )
    return kwargs


def reask_gemini_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Gemini
    exception: Exception,
):
    from google.ai import generativelanguage as glm  # type: ignore

    reask_msgs = [
        {
            "role": "function",
            "parts": [
                glm.Part(
                    function_response=glm.FunctionResponse(
                        name=response.parts[0].function_call.name,
                        response={"error": f"Validation Error(s) found:\n{exception}"},
                    )
                ),
            ],
        },
        {
            "role": "user",
            "parts": ["Recall the function arguments correctly and fix the errors"],
        },
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_gemini_json(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Gemini
    exception: Exception,
):
    kwargs["contents"].append(
        {
            "role": "user",
            "parts": [
                f"Correct the following JSON response, based on the errors given below:\n\n"
                f"JSON:\n{response.text}\n\nExceptions:\n{exception}"
            ],
        }
    )
    return kwargs


def reask_vertexai_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Vertex AI
    exception: Exception,
):
    from .client_vertexai import vertexai_function_response_parser

    kwargs = kwargs.copy()
    reask_msgs = [
        response.candidates[0].content,
        vertexai_function_response_parser(response, exception),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_vertexai_json(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Vertex AI
    exception: Exception,
):
    from .client_vertexai import vertexai_message_parser

    kwargs = kwargs.copy()

    reask_msgs = [
        response.candidates[0].content,
        vertexai_message_parser(
            {
                "role": "user",
                "content": (
                    f"Validation Errors found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors found in the following attempt:\n{response.text}"
                ),
            }
        ),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "tool",  # type: ignore
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_cerebras_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors and call the tool {tool_call.function.name} again, "
                    f"taking into account the problems with {tool_call.function.arguments} that was previously generated."
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_md_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_default(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Recall the function correctly, fix the errors, exceptions found\n{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_fireworks_tools(kwargs: dict[str, Any], response: Any, exception: Exception):
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "tool",  # type: ignore
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_fireworks_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_reask_kwargs(
    kwargs: dict[str, Any],
    mode: Mode,
    response: Any,  # Replace with actual response type based on the mode
    exception: Exception,
):
    kwargs = kwargs.copy()

    functions = {
        Mode.ANTHROPIC_TOOLS: reask_anthropic_tools,
        Mode.ANTHROPIC_JSON: reask_anthropic_json,
        Mode.COHERE_TOOLS: reask_cohere_tools,
        Mode.COHERE_JSON_SCHEMA: reask_cohere_tools,  # Same Function
        Mode.GEMINI_TOOLS: reask_gemini_tools,
        Mode.GEMINI_JSON: reask_gemini_json,
        Mode.VERTEXAI_TOOLS: reask_vertexai_tools,
        Mode.VERTEXAI_JSON: reask_vertexai_json,
        Mode.TOOLS: reask_tools,
        Mode.TOOLS_STRICT: reask_tools,
        Mode.CEREBRAS_TOOLS: reask_cerebras_tools,
        Mode.MD_JSON: reask_md_json,
        Mode.FIREWORKS_TOOLS: reask_fireworks_tools,
        Mode.FIREWORKS_JSON: reask_fireworks_json,
    }
    reask_function = functions.get(mode, reask_default)
    return reask_function(kwargs=kwargs, response=response, exception=exception)

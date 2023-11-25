import inspect
from functools import wraps
from instructor.dsl.multitask import MultiTask, MultiTaskBase
from json import JSONDecodeError
from typing import get_origin, get_args, Callable, Optional, Type, Union
from collections.abc import Iterable

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from .function_calls import OpenAISchema, openai_schema, Mode

import warnings

OVERRIDE_DOCS = """
Creates a new chat completion for the provided messages and parameters.

See: https://platform.openai.com/docs/api-reference/chat-completions/create

Additional Notes:

Using the `response_model` parameter, you can specify a response model to use for parsing the response from OpenAI's API. If its present, the response will be parsed using the response model, otherwise it will be returned as is. 

If `stream=True` is specified, the response will be parsed using the `from_stream_response` method of the response model, if available, otherwise it will be parsed using the `from_response` method.

If need to obtain the raw response from OpenAI's API, you can access it using the `_raw_response` attribute of the response model.

Parameters:
    response_model (Union[Type[BaseModel], Type[OpenAISchema]]): The response model to use for parsing the response from OpenAI's API, if available (default: None)
    max_retries (int): The maximum number of retries to attempt if the response is not valid (default: 0)
    validation_context (dict): The validation context to use for validating the response (default: None)
"""


def dump_message(message) -> dict:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """
    dumped_message = message.model_dump()
    if not dumped_message.get("tool_calls"):
        del dumped_message["tool_calls"]
    return {k: v for k, v in dumped_message.items() if v}


def handle_response_model(
    *,
    response_model: Type[BaseModel],
    kwargs,
    mode: Mode = Mode.FUNCTIONS,
):
    new_kwargs = kwargs.copy()
    if response_model is not None:
        if get_origin(response_model) is Iterable:
            iterable_element_class = get_args(response_model)[0]
            response_model = MultiTask(iterable_element_class)
        if not issubclass(response_model, OpenAISchema):
            response_model = openai_schema(response_model)  # type: ignore
        
        if new_kwargs.get("stream", False) and not issubclass(response_model, MultiTaskBase):
            raise NotImplementedError("stream=True is not supported when using response_model parameter for non-iterables")

            warnings.warn(
                "stream=True is not supported when using response_model parameter for non-iterables"
            )
        if mode == Mode.FUNCTIONS:
            new_kwargs["functions"] = [response_model.openai_schema]  # type: ignore
            new_kwargs["function_call"] = {
                "name": response_model.openai_schema["name"]
            }  # type: ignore
        elif mode == Mode.TOOLS:
            new_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": response_model.openai_schema,
                }
            ]
            new_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": response_model.openai_schema["name"]},
            }
        elif mode == Mode.JSON:
            new_kwargs["response_format"] = {"type": "json_object"}

            # check that the first message is a system message
            # if it is not, add a system message to the beginning
            message = f"Make sure that your response to any message matchs the json_schema below, do not deviate at all: \n{response_model.model_json_schema()['properties']}"

            if new_kwargs["messages"][0]["role"] != "system":
                new_kwargs["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": message,
                    },
                )

            # if the first message is a system append the schema to the end
            if new_kwargs["messages"][0]["role"] == "system":
                new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        else:
            raise ValueError(f"Invalid patch mode: {mode}")


    return response_model, new_kwargs


def process_response(
    response,
    *,
    response_model: Type[BaseModel],
    validation_context: dict = None,
    strict=None,
    stream=False,
    mode: Mode = Mode.FUNCTIONS,
):  # type: ignore
    """Processes a OpenAI response with the response model, if available.
    It can use `validation_context` and `strict` to validate the response
    via the pydantic model

    Args:
        response (ChatCompletion): The response from OpenAI's API
        response_model (BaseModel): The response model to use for parsing the response
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (bool, optional): Whether to use strict json parsing. Defaults to None.
    """
    if response_model is not None:
        stream_multitask = stream and issubclass(response_model, MultiTaskBase)
        model = response_model.from_response(
            response, validation_context=validation_context, strict=strict, mode=mode, stream_multitask=stream_multitask
        )
        if not stream:
            model._raw_response = response
        return model
    return response


async def retry_async(
    func,
    response_model,
    validation_context,
    args,
    kwargs,
    max_retries,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
):
    retries = 0
    while retries <= max_retries:
        try:
            response: ChatCompletion = await func(*args, **kwargs)
            stream = kwargs.get("stream", False)
            return process_response(
                response,
                response_model=response_model,
                validation_context=validation_context,
                strict=strict,
                mode=mode,
                stream=stream,
            )
        except (ValidationError, JSONDecodeError) as e:
            kwargs["messages"].append(dump_message(response.choices[0].message))  # type: ignore
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": f"Recall the function correctly, exceptions found\n{e}",
                }
            )
            retries += 1
            if retries > max_retries:
                raise e


def retry_sync(
    func,
    response_model,
    validation_context,
    args,
    kwargs,
    max_retries,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
):
    retries = 0
    while retries <= max_retries:
        # Excepts ValidationError, and JSONDecodeError
        try:
            response = func(*args, **kwargs)
            stream = kwargs.get("stream", False)
            return process_response(
                response,
                response_model=response_model,
                validation_context=validation_context,
                strict=strict,
                mode=mode,
                stream=stream
            )
        except (ValidationError, JSONDecodeError) as e:
            kwargs["messages"].append(response.choices[0].message)
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": f"Recall the function correctly, exceptions found\n{e}",
                }
            )
            retries += 1
            if retries > max_retries:
                raise e


def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    return inspect.iscoroutinefunction(func) or (
        hasattr(func, "__wrapped__") and inspect.iscoroutinefunction(func.__wrapped__)
    )


def wrap_chatcompletion(
    func: Callable, mode: Mode = Mode.FUNCTIONS
) -> Callable:
    func_is_async = is_async(func)

    @wraps(func)
    async def new_chatcompletion_async(
        response_model=None,
        validation_context=None,
        max_retries=1,
        *args,
        **kwargs,
    ):
        if mode == Mode.TOOLS:
            max_retries = 0
            warnings.warn("max_retries is not supported when using tool calls")

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, kwargs=kwargs, mode=mode
        )  # type: ignore
        response = await retry_async(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )  # type: ignore
        return response

    @wraps(func)
    def new_chatcompletion_sync(
        response_model=None,
        validation_context=None,
        max_retries=1,
        *args,
        **kwargs,
    ):
        if mode == Mode.TOOLS:
            max_retries = 0
            warnings.warn("max_retries is not supported when using tool calls")

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, kwargs=kwargs, mode=mode
        )  # type: ignore
        response = retry_sync(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )  # type: ignore
        return response

    wrapper_function = (
        new_chatcompletion_async if func_is_async else new_chatcompletion_sync
    )
    wrapper_function.__doc__ = OVERRIDE_DOCS
    return wrapper_function


def patch(
    client: Union[OpenAI, AsyncOpenAI], mode: Mode = Mode.FUNCTIONS
):
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    client.chat.completions.create = wrap_chatcompletion(
        client.chat.completions.create, mode=mode
    )
    return client


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
    return patch(client, mode=mode)

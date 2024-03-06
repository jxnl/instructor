# type: ignore[all]
import inspect
import logging
from textwrap import dedent
from collections.abc import Iterable
from functools import wraps
from tenacity import Retrying, AsyncRetrying, stop_after_attempt, RetryError
from json import JSONDecodeError
from typing import (
    Callable,
    Generator,
    Optional,
    ParamSpec,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError

from instructor.dsl.iterable import IterableModel, IterableBase
from instructor.dsl.parallel import ParallelBase, ParallelModel, handle_parallel_model
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import ModelAdapter, AdapterBase, is_simple_type
from instructor.utils import dump_message, update_total_usage

from .function_calls import Mode, OpenAISchema, openai_schema

logger = logging.getLogger("instructor")
T = TypeVar("T")


T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def handle_response_model(
    response_model: T, mode: Mode = Mode.TOOLS, **kwargs
) -> Union[Type[OpenAISchema], dict]:
    """Prepare the response model type hint, and returns the response_model
    along with the new modified kwargs needed to be able to use the response_model
    parameter with the patch function.


    Args:
        response_model (T): The response model to use for parsing the response
        mode (Mode, optional): The openai completion mode. Defaults to Mode.TOOLS.

    Raises:
        NotImplementedError: When using stream=True with a non-iterable response_model
        ValueError: When using an invalid patch mode

    Returns:
        Union[Type[OpenAISchema], dict]: The response model to use for parsing the response
    """
    new_kwargs = kwargs.copy()
    if response_model is not None:
        # Handles the case where the response_model is a simple type
        # Literal, Annotated, Union, str, int, float, bool, Enum
        # We wrap the response_model in a ModelAdapter that sets 'content' as the response
        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]

        # This a special case for parallel tools
        if mode == Mode.PARALLEL_TOOLS:
            assert (
                new_kwargs.get("stream", False) is False
            ), "stream=True is not supported when using PARALLEL_TOOLS mode"
            new_kwargs["tools"] = handle_parallel_model(response_model)
            new_kwargs["tool_choice"] = "auto"

            # This is a special case for parallel models
            response_model = ParallelModel(typehint=response_model)
            return response_model, new_kwargs

        # This is for all other single model cases
        if get_origin(response_model) is Iterable:
            iterable_element_class = get_args(response_model)[0]
            response_model = IterableModel(iterable_element_class)
        if not issubclass(response_model, OpenAISchema):
            response_model = openai_schema(response_model)  # type: ignore

        if new_kwargs.get("stream", False) and not issubclass(
            response_model, (IterableBase, PartialBase)
        ):
            raise NotImplementedError(
                "stream=True is not supported when using response_model parameter for non-iterables"
            )

        if mode == Mode.ANTHROPIC:
            """TODO: We should assume that we've only patched the create call and not the client itself. 

            new_kwargs["messages"] += "\n"
            new_kwargs["messages"] += response_model.anthropic_schema
            """
            raise NotImplementedError

        if mode == Mode.FUNCTIONS:
            new_kwargs["functions"] = [response_model.openai_schema]  # type: ignore
            new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}  # type: ignore
        elif mode in {Mode.TOOLS, Mode.MISTRAL_TOOLS}:
            new_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": response_model.openai_schema,
                }
            ]
            if mode == Mode.MISTRAL_TOOLS:
                new_kwargs["tool_choice"] = "any"
            else:
                new_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": response_model.openai_schema["name"]},
                }
        elif mode in {Mode.JSON, Mode.MD_JSON, Mode.JSON_SCHEMA}:
            # If its a JSON Mode we need to massage the prompt a bit
            # in order to get the response we want in a json format
            message = dedent(
                f"""
                As a genius expert, your task is to understand the content and provide
                the parsed objects in json that match the following json_schema:\n

                {response_model.model_json_schema()}

                Make sure to return an instance of the JSON, not the schema itself
                """
            )

            if mode == Mode.JSON:
                new_kwargs["response_format"] = {"type": "json_object"}

            elif mode == Mode.JSON_SCHEMA:
                new_kwargs["response_format"] = {
                    "type": "json_object",
                    "schema": response_model.model_json_schema(),
                }

            elif mode == Mode.MD_JSON:
                new_kwargs["messages"].append(
                    {
                        "role": "user",
                        "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
                    },
                )
            # check that the first message is a system message
            # if it is not, add a system message to the beginning
            if new_kwargs["messages"][0]["role"] != "system":
                new_kwargs["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": message,
                    },
                )
            # if it is, system append the schema to the end
            else:
                new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        else:
            raise ValueError(f"Invalid patch mode: {mode}")

    logger.debug(
        f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}",
        extra={
            "mode": mode.value,
            "response_model": response_model.__name__
            if response_model is not None
            else None,
            "new_kwargs": new_kwargs,
        },
    )
    return response_model, new_kwargs


def process_response(
    response: T_Model,
    *,
    response_model: Type[OpenAISchema | BaseModel],
    stream: bool,
    validation_context: Optional[dict] = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | Generator[T_Model, None, None]:
    """Processes a OpenAI response with the response model, if available.

    Args:
        response (T): The response from OpenAI's API
        response_model (Type[T_Model]): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (_type_, optional): Whether to use strict json parsing. Defaults to None.
        mode (Mode, optional): The openai completion mode. Defaults to Mode.FUNCTIONS.

    Returns:
        Union[T_Model, T]: The parsed response, if a response model is available, otherwise the response as is from the SDK
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )

    if response_model is None:
        logger.debug("No response model, returning response as is")
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = response_model.from_streaming_response(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: Type[T_Model | OpenAISchema | BaseModel],
    stream: bool = False,
    validation_context: Optional[dict] = None,
    strict: Optional[bool] = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | ChatCompletion:
    """Processes a OpenAI response with the response model, if available.
    It can use `validation_context` and `strict` to validate the response
    via the pydantic model

    Args:
        response (ChatCompletion): The response from OpenAI's API
        response_model (BaseModel): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (bool, optional): Whether to use strict json parsing. Defaults to None.
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )
    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = await response_model.from_streaming_response_async(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


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
                    response: ChatCompletion = await func(*args, **kwargs)  # type: ignore
                    stream = kwargs.get("stream", False)
                    response = update_total_usage(response, total_usage)
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
                                "role": "user",
                                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
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
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e


def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine


class InstructorChatCompletionCreate(Protocol):
    def __call__(
        self,
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        ...


@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.TOOLS,
) -> OpenAI:
    ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
) -> AsyncOpenAI:
    ...


@overload
def patch(
    create: Callable[T_ParamSpec, T_Retval],
    mode: Mode = Mode.TOOLS,
) -> InstructorChatCompletionCreate:
    ...


def patch(
    client: Union[OpenAI, AsyncOpenAI] = None,
    create: Callable[T_ParamSpec, T_Retval] = None,
    mode: Mode = Mode.TOOLS,
) -> Union[OpenAI, AsyncOpenAI]:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if create is not None:
        func = create
    elif client is not None:
        func = client.chat.completions.create
    else:
        raise ValueError("Either client or create must be provided")

    func_is_async = is_async(func)

    @wraps(func)
    async def new_create_async(
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )
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
    def new_create_sync(
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )
        response = retry_sync(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )
        return response

    new_create = new_create_async if func_is_async else new_create_sync

    if client is not None:
        client.chat.completions.create = new_create
        return client
    else:
        return new_create


def apatch(client: AsyncOpenAI, mode: Mode = Mode.TOOLS):
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

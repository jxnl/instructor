from __future__ import annotations

import openai
import inspect
import instructor
from .utils import Provider, get_provider
from openai.types.chat import ChatCompletionMessageParam
from typing import (
    TypeVar,
    Callable,
    overload,
    Union,
    Any,
)
from collections.abc import Generator, Iterable, Awaitable, AsyncGenerator
from typing_extensions import Self
from pydantic import BaseModel
from instructor.dsl.partial import Partial
import logfire_api as logfire


T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class Instructor:
    client: Any | None
    create_fn: Callable[..., Any]
    mode: instructor.Mode
    default_model: str | None = None
    provider: Provider

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        if mode == instructor.Mode.FUNCTIONS:
            instructor.Mode.warn_mode_functions_deprecation()
        self.kwargs = kwargs
        self.provider = provider

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    @property
    def messages(self) -> Self:
        return self

    @overload
    def create(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Awaitable[T]: ...

    @overload
    def create(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self: AsyncInstructor,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Awaitable[Any]: ...

    @overload
    def create(
        self: Self,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Any: ...

    def create(
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T | Any | Awaitable[T] | Awaitable[Any]:
        with logfire.span(
            "Creating Chat Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs = self.handle_kwargs(kwargs)

            result = self.create_fn(
                response_model=response_model,
                messages=messages,
                max_retries=max_retries,
                validation_context=validation_context,
                strict=strict,
                **kwargs,
            )
            logfire.info("Result", result=result)  # type:ignore
            return result

    @overload
    def create_partial(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_partial(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_partial(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        with logfire.span(
            "Creating Streaming Chat Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs["stream"] = True

            kwargs = self.handle_kwargs(kwargs)

            response_model = instructor.Partial[response_model]  # type: ignore
            return self.create_fn(
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
                validation_context=validation_context,
                strict=strict,
                **kwargs,
            )

    @overload
    def create_iterable(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_iterable(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_iterable(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        with logfire.span(
            "Creating Iterable Chat Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs["stream"] = True
            kwargs = self.handle_kwargs(kwargs)

            response_model = Iterable[response_model]  # type: ignore
            return self.create_fn(
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
                validation_context=validation_context,
                strict=strict,
                **kwargs,
            )

    @overload
    def create_with_completion(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Awaitable[tuple[T, Any]]: ...

    @overload
    def create_with_completion(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[T, Any]: ...

    def create_with_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[T, Any] | Awaitable[tuple[T, Any]]:
        with logfire.span(
            "Creating Chat Completion with Original Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs = self.handle_kwargs(kwargs)
            model = self.create_fn(
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
                validation_context=validation_context,
                strict=strict,
                **kwargs,
            )
            return model, model._raw_response

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs


class AsyncInstructor(Instructor):
    client: Any | None
    create_fn: Callable[..., Any]
    mode: instructor.Mode
    default_model: str | None = None
    provider: Provider

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        self.kwargs = kwargs
        self.provider = provider

    async def create(
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T | Any:
        with logfire.span(
            "Creating Async Chat Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs = self.handle_kwargs(kwargs)
            result = await self.create_fn(
                response_model=response_model,
                validation_context=validation_context,
                max_retries=max_retries,
                messages=messages,
                strict=strict,
                **kwargs,
            )
            logfire.info("Result", result=result)  # type:ignore
            return result

    async def create_partial(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        with logfire.span(
            "Creating Async Partial Chat Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs = self.handle_kwargs(kwargs)
            kwargs["stream"] = True
            final_result = None
            async for item in await self.create_fn(
                response_model=instructor.Partial[response_model],  # type: ignore
                validation_context=validation_context,
                max_retries=max_retries,
                messages=messages,
                strict=strict,
                **kwargs,
            ):
                final_result = item
                yield item
            logfire.info("Iterable Result", item=final_result)  # type:ignore

    async def create_iterable(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        with logfire.span(
            "Creating Async Iterable Chat Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs = self.handle_kwargs(kwargs)
            kwargs["stream"] = True
            async for item in await self.create_fn(
                response_model=Iterable[response_model],
                validation_context=validation_context,
                max_retries=max_retries,
                messages=messages,
                strict=strict,
                **kwargs,
            ):
                logfire.info("Iterable Generation Result", item=item)  # type:ignore
                yield item

    async def create_with_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        with logfire.span(
            "Creating Async Chat Completion with Original Completion",
            response_model=response_model.model_json_schema()  # type:ignore
            if response_model
            else None,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            strict=strict,
            **kwargs,
        ):
            kwargs = self.handle_kwargs(kwargs)
            response = await self.create_fn(
                response_model=response_model,
                validation_context=validation_context,
                max_retries=max_retries,
                messages=messages,
                strict=strict,
                **kwargs,
            )
            logfire.info(  # type:ignore
                "Result", result=response, original_completion=response._raw_response
            )

            return response, response._raw_response


@overload
def from_openai(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor:
    pass


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor:
    pass


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    if hasattr(client, "base_url"):
        provider = get_provider(str(client.base_url))
    else:
        provider = Provider.OPENAI

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        import warnings

        warnings.warn(
            "Client should be an instance of openai.OpenAI or openai.AsyncOpenAI. Unexpected behavior may occur with other client types.",
            stacklevel=2,
        )

    if provider in {Provider.ANYSCALE, Provider.TOGETHER}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.JSON_SCHEMA,
            instructor.Mode.MD_JSON,
        }

    if provider in {Provider.DATABRICKS}:
        assert mode in {
            instructor.Mode.MD_JSON
        }, "Databricks provider only supports `MD_JSON` mode."

    if provider in {Provider.OPENAI}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.FUNCTIONS,
            instructor.Mode.PARALLEL_TOOLS,
            instructor.Mode.MD_JSON,
        }

    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            mode=mode,
            provider=provider,
            **kwargs,
        )

    if isinstance(client, openai.AsyncOpenAI):
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            mode=mode,
            provider=provider,
            **kwargs,
        )


@overload
def from_litellm(
    completion: Callable[..., Any],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_litellm(
    completion: Awaitable[Any],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor:
    pass


def from_litellm(
    completion: Callable[..., Any] | Awaitable[Any],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    is_async = inspect.iscoroutinefunction(completion)

    if not is_async:
        return Instructor(
            client=None,
            create=instructor.patch(create=completion, mode=mode),
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=None,
            create=instructor.patch(create=completion, mode=mode),
            mode=mode,
            **kwargs,
        )

# Modified from src/openai/resources/chat/completions.py

from __future__ import annotations

from typing import Any, Dict, List, Type, TypeVar, Union, Iterable, Optional, overload
from typing_extensions import Literal

import httpx
from openai import AsyncStream

from openai._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from openai._streaming import Stream
from openai.resources.chat import Completions, AsyncCompletions
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
from pydantic import BaseModel

from instructor.response import handle_response_model
from instructor.retry import retry_async, retry_sync
from instructor.function_calls import Mode

Model = (
    str
    | Literal[
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k-0613",
    ]
)


class InstructorOpenAIChatCompletions(Completions):
    T_CompletionResponseModel = TypeVar("T_CompletionResponseModel", bound=BaseModel)

    def __init__(self, openai_completions: Completions, mode=Mode.FUNCTIONS) -> None:
        self.__dict__.update(openai_completions.__dict__)
        self._openai_completions: Completions = openai_completions
        self._mode = mode

    @overload
    def create(
        self,
        *,
        max_retries: int = 1,
        messages: Iterable[ChatCompletionMessageParam],
        model: Model,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
        ...

    @overload
    def create(
        self,
        *,
        response_model: Type[T_CompletionResponseModel],
        messages: Iterable[ChatCompletionMessageParam],
        model: Model,
        validation_context: dict | None = None,
        max_retries: int = 1,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> T_CompletionResponseModel:
        ...

    # TODO - stream signatures

    # @overload
    # def create(
    #     self,
    #     *,
    #     messages: Iterable[ChatCompletionMessageParam],
    #     model: Model,
    #     stream: Literal[True],
    #     response_model: Type[T_CompletionIterableModel],
    #     validation_context: dict | None = None,
    #     max_retries: int = 1,
    #     frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    #     logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    #     max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    #     n: Optional[int] | NotGiven = NOT_GIVEN,
    #     presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
    #     seed: Optional[int] | NotGiven = NOT_GIVEN,
    #     stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    #     temperature: Optional[float] | NotGiven = NOT_GIVEN,
    #     top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    #     top_p: Optional[float] | NotGiven = NOT_GIVEN,
    #     user: str | NotGiven = NOT_GIVEN,
    #     # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    #     # The extra values given here take precedence over values defined on the client or passed to this method.
    #     extra_headers: Headers | None = None,
    #     extra_query: Query | None = None,
    #     extra_body: Body | None = None,
    #     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    # ) -> T_CompletionIterableModel:
    #     ...

    # @overload
    # def create(
    #     self,
    #     *,
    #     messages: Iterable[ChatCompletionMessageParam],
    #     model: Model,
    #     stream: Literal[True],
    #     max_retries: int = 1,
    #     frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
    #     functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
    #     logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    #     logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    #     max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    #     n: Optional[int] | NotGiven = NOT_GIVEN,
    #     presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
    #     seed: Optional[int] | NotGiven = NOT_GIVEN,
    #     stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    #     temperature: Optional[float] | NotGiven = NOT_GIVEN,
    #     tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    #     tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
    #     top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    #     top_p: Optional[float] | NotGiven = NOT_GIVEN,
    #     user: str | NotGiven = NOT_GIVEN,
    #     # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    #     # The extra values given here take precedence over values defined on the client or passed to this method.
    #     extra_headers: Headers | None = None,
    #     extra_query: Query | None = None,
    #     extra_body: Body | None = None,
    #     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    # ) -> Stream[ChatCompletionChunk]:
    #     ...

    def create(
        self,
        *args,
        response_model: Type[T_CompletionResponseModel] | None = None,
        validation_context: dict | None = None,
        max_retries: int = 1,
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletionChunk] | T_CompletionResponseModel | Any:
        # args = [
        #     messages,
        #     model,
        # ]

        # kwargs = {
        #     "stream": stream,
        #     "frequency_penalty": frequency_penalty,
        #     "function_call": function_call,
        #     "functions": functions,
        #     "logit_bias": logit_bias,
        #     "logprobs": logprobs,
        #     "max_tokens": max_tokens,
        #     "n": n,
        #     "presence_penalty": presence_penalty,
        #     "response_format": response_format,
        #     "seed": seed,
        #     "stop": stop,
        #     "temperature": temperature,
        #     "tool_choice": tool_choice,
        #     "tools": tools,
        #     "top_logprobs": top_logprobs,
        #     "top_p": top_p,
        #     "user": user,
        #     "extra_headers": extra_headers,
        #     "extra_query": extra_query,
        #     "extra_body": extra_body,
        #     "timeout": timeout,
        # }

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=self._mode, **kwargs
        )

        response = retry_sync(
            func=self._openai_completions.create,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=self._mode,
        )

        return response


class InstructorAsyncOpenAIChatCompletions(AsyncCompletions):
    T_CompletionResponseModel = TypeVar("T_CompletionResponseModel", bound=BaseModel)

    def __init__(self, openai_completions: Completions, mode=Mode.FUNCTIONS) -> None:
        self.__dict__.update(openai_completions.__dict__)
        self._openai_completions: Completions = openai_completions
        self._mode = mode

    @overload
    async def create(
        self,
        *,
        max_retries: int = 1,
        messages: Iterable[ChatCompletionMessageParam],
        model: Model,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
        ...

    @overload
    async def create(
        self,
        *,
        response_model: Type[T_CompletionResponseModel],
        messages: Iterable[ChatCompletionMessageParam],
        model: Model,
        validation_context: dict | None = None,
        max_retries: int = 1,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> T_CompletionResponseModel:
        ...

    # TODO - stream signatures

    # @overload
    # def create(
    #     self,
    #     *,
    #     messages: Iterable[ChatCompletionMessageParam],
    #     model: Model,
    #     stream: Literal[True],
    #     response_model: Type[T_CompletionIterableModel],
    #     validation_context: dict | None = None,
    #     max_retries: int = 1,
    #     frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    #     logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    #     max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    #     n: Optional[int] | NotGiven = NOT_GIVEN,
    #     presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
    #     seed: Optional[int] | NotGiven = NOT_GIVEN,
    #     stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    #     temperature: Optional[float] | NotGiven = NOT_GIVEN,
    #     top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    #     top_p: Optional[float] | NotGiven = NOT_GIVEN,
    #     user: str | NotGiven = NOT_GIVEN,
    #     # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    #     # The extra values given here take precedence over values defined on the client or passed to this method.
    #     extra_headers: Headers | None = None,
    #     extra_query: Query | None = None,
    #     extra_body: Body | None = None,
    #     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    # ) -> T_CompletionIterableModel:
    #     ...

    # @overload
    # def create(
    #     self,
    #     *,
    #     messages: Iterable[ChatCompletionMessageParam],
    #     model: Model,
    #     stream: Literal[True],
    #     max_retries: int = 1,
    #     frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
    #     functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
    #     logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    #     logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    #     max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    #     n: Optional[int] | NotGiven = NOT_GIVEN,
    #     presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    #     response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
    #     seed: Optional[int] | NotGiven = NOT_GIVEN,
    #     stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    #     temperature: Optional[float] | NotGiven = NOT_GIVEN,
    #     tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    #     tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
    #     top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    #     top_p: Optional[float] | NotGiven = NOT_GIVEN,
    #     user: str | NotGiven = NOT_GIVEN,
    #     # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    #     # The extra values given here take precedence over values defined on the client or passed to this method.
    #     extra_headers: Headers | None = None,
    #     extra_query: Query | None = None,
    #     extra_body: Body | None = None,
    #     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    # ) -> Stream[ChatCompletionChunk]:
    #     ...

    async def create(
        self,
        *args,
        response_model: Type[T_CompletionResponseModel] | None = None,
        validation_context: dict | None = None,
        max_retries: int = 1,
        **kwargs,
    ) -> (
        ChatCompletion
        | AsyncStream[ChatCompletionChunk]
        | T_CompletionResponseModel
        | Any
    ):
        # args = [
        #     messages,
        #     model,
        # ]

        # kwargs = {
        #     "stream": stream,
        #     "frequency_penalty": frequency_penalty,
        #     "function_call": function_call,
        #     "functions": functions,
        #     "logit_bias": logit_bias,
        #     "logprobs": logprobs,
        #     "max_tokens": max_tokens,
        #     "n": n,
        #     "presence_penalty": presence_penalty,
        #     "response_format": response_format,
        #     "seed": seed,
        #     "stop": stop,
        #     "temperature": temperature,
        #     "tool_choice": tool_choice,
        #     "tools": tools,
        #     "top_logprobs": top_logprobs,
        #     "top_p": top_p,
        #     "user": user,
        #     "extra_headers": extra_headers,
        #     "extra_query": extra_query,
        #     "extra_body": extra_body,
        #     "timeout": timeout,
        # }

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=self._mode, **kwargs
        )

        response = await retry_async(
            func=self._openai_completions.create,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=self._mode,
        )

        return response

from typing import Literal, Any, Union, TypeVar, get_args
from collections.abc import Iterable
from pydantic import BaseModel, Field
from instructor.mode import Mode
from instructor.process_response import handle_response_model
import uuid
import json

T = TypeVar("T", bound=BaseModel)

openai_models = Literal[
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
]
anthropic_models = Literal[
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620",
]


class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: Union[openai_models, anthropic_models, str]
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=1.0)
    tools: Union[list[Tool], list[dict[str, Any]]]
    tool_choice: dict[str, Any]


class BatchModel(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/chat/completions"]
    body: RequestBody


class AnthropicBatchModel(BaseModel):
    custom_id: str
    params: RequestBody


class BatchJob:
    @classmethod
    def parse_from_file(
        cls,
        file_path: str,
        response_model: type[T],
        model: Union[openai_models, anthropic_models, str],
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        with open(file_path) as file:
            res, error_objs = [], []
            for line in file:
                data = json.loads(line)
                try:
                    args = (
                        data["response"]["body"]["choices"][0]["message"]["tool_calls"][
                            0
                        ]["function"]["arguments"]
                        if model in get_args(openai_models)
                        else data["result"]["message"]["content"][0]["input"]
                    )
                    res.append(response_model(**args))
                except Exception:
                    error_objs.append(data)
            return res, error_objs

    @classmethod
    def parse_from_string(
        cls,
        content: str,
        response_model: type[T],
        model: Union[openai_models, anthropic_models, str],
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        res, error_objs = [], []
        for line in content.splitlines():
            data = json.loads(line)
            try:
                args = (
                    data["response"]["body"]["choices"][0]["message"]["tool_calls"][0][
                        "function"
                    ]["arguments"]
                    if model in get_args(openai_models)
                    else data["result"]["message"]["content"][0]["input"]
                )
                res.append(response_model(**args))
            except Exception:
                error_objs.append(data)
        return res, error_objs

    @classmethod
    def create_from_messages(
        cls,
        messages_batch: Union[
            list[list[dict[str, Any]]], Iterable[list[dict[str, Any]]]
        ],
        model: Union[openai_models, str],
        response_model: type[BaseModel],
        file_path: str,
        max_tokens: int = 1000,
        temperature: float = 1.0,
    ):
        _, kwargs = handle_response_model(response_model=response_model)

        with open(file_path, "w") as file:
            for messages in messages_batch:
                if model in get_args(anthropic_models):
                    _, kwargs = handle_response_model(
                        response_model=response_model,
                        mode=Mode.ANTHROPIC_TOOLS,
                        messages=messages,
                    )
                    _ = kwargs.pop("messages")

                request_body = RequestBody(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )
                if model in get_args(openai_models):
                    batch_request = BatchModel(
                        custom_id=str(uuid.uuid4()),
                        method="POST",
                        url="/v1/chat/completions",
                        body=request_body,
                    ).model_dump_json()

                elif model in get_args(anthropic_models):
                    batch_request = AnthropicBatchModel(
                        custom_id=str(uuid.uuid4()),
                        params=request_body,
                    ).model_dump_json()
                else:
                    raise ValueError(
                        f"Batch processing for model {model} not supported"
                    )

                file.write(batch_request + "\n")

from __future__ import annotations

from typing import Any

from vertexai.preview.generative_models import ToolConfig  # type: ignore
import vertexai.generative_models as gm  # type: ignore
from pydantic import BaseModel
import instructor
import jsonref  # type: ignore


def _create_gemini_json_schema(model: BaseModel):
    schema = model.model_json_schema()
    schema_without_refs: dict[str, Any] = jsonref.replace_refs(schema)  # type: ignore
    try:
        required_properties = schema_without_refs["required"]
    except KeyError:
        required_properties = []
    # Find properties that have "default" value set, and add them to `required_properties`.
    # There shouldn't be any duplicates because properties with "default" are not required.
    required_properties += [prop for prop, description in schema_without_refs["properties"].items() if "default" in description]
    gemini_schema: dict[Any, Any] = {
        "type": schema_without_refs["type"],
        "properties": schema_without_refs["properties"],
        "required": required_properties,
    }
    return gemini_schema


def _create_vertexai_tool(model: BaseModel) -> gm.Tool:
    parameters = _create_gemini_json_schema(model)

    declaration = gm.FunctionDeclaration(
        name=model.__name__, description=model.__doc__, parameters=parameters
    )

    tool = gm.Tool(function_declarations=[declaration])

    return tool


def vertexai_message_parser(message: dict[str, str]) -> gm.Content:
    return gm.Content(
        role=message["role"], parts=[gm.Part.from_text(message["content"])]
    )


def _vertexai_message_list_parser(messages: list[dict[str, str]]) -> list[gm.Content]:
    contents = [
        vertexai_message_parser(message) if isinstance(message, dict) else message
        for message in messages
    ]
    return contents


def vertexai_function_response_parser(
    response: gm.GenerationResponse, exception: Exception
) -> gm.Content:
    return gm.Content(
        parts=[
            gm.Part.from_function_response(
                name=response.candidates[0].content.parts[0].function_call.name,
                response={
                    "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                },
            )
        ]
    )


def vertexai_process_response(_kwargs: dict[str, Any], model: BaseModel):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)

    tool = _create_vertexai_tool(model=model)

    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        )
    )
    return contents, [tool], tool_config


def vertexai_process_json_response(_kwargs: dict[str, Any], model: BaseModel):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)

    config: dict[str, Any] | None = _kwargs.pop("generation_config", None)

    response_schema = _create_gemini_json_schema(model)

    generation_config = gm.GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        **(config if config else {}),
    )

    return contents, generation_config


def from_vertexai(
    client: gm.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.VERTEXAI_TOOLS,
    _async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    assert mode in {
        instructor.Mode.VERTEXAI_TOOLS,
        instructor.Mode.VERTEXAI_JSON,
    }, "Mode must be instructor.Mode.VERTEXAI_TOOLS"

    assert isinstance(
        client, gm.GenerativeModel
    ), "Client must be an instance of vertexai.generative_models.GenerativeModel"

    create = client.generate_content_async if _async else client.generate_content

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.VERTEXAI,
        mode=mode,
        **kwargs,
    )

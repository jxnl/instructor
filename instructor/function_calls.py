# type: ignore
import json
import logging
from functools import wraps
from typing import Annotated, Any, Optional, TypeVar, cast

from docstring_parser import parse
from openai.types.chat import ChatCompletion
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
)

from instructor.exceptions import IncompleteOutputException
from instructor.mode import Mode
from instructor.utils import (
    classproperty,
    extract_json_from_codeblock,
    map_to_gemini_function_schema,
)

T = TypeVar("T")

logger = logging.getLogger("instructor")


class OpenAISchema(BaseModel):
    # Ignore classproperty, since Pydantic doesn't understand it like it would a normal property.
    model_config = ConfigDict(ignored_types=(classproperty,))

    @classproperty
    def openai_schema(cls) -> dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classproperty
    def anthropic_schema(cls) -> dict[str, Any]:
        return {
            "name": cls.openai_schema["name"],
            "description": cls.openai_schema["description"],
            "input_schema": cls.model_json_schema(),
        }

    @classproperty
    def gemini_schema(cls) -> Any:
        import google.generativeai.types as genai_types

        function = genai_types.FunctionDeclaration(
            name=cls.openai_schema["name"],
            description=cls.openai_schema["description"],
            parameters=map_to_gemini_function_schema(cls.openai_schema["parameters"]),
        )
        return function

    @classmethod
    def from_response(
        cls,
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
        mode: Mode = Mode.TOOLS,
    ) -> BaseModel:
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            throw_error (bool): Whether to throw an error if the function call is not detected
            validation_context (dict): The validation context to use for validating the response
            strict (bool): Whether to use strict json parsing
            mode (Mode): The openai completion mode

        Returns:
            cls (OpenAISchema): An instance of the class
        """
        if mode == Mode.ANTHROPIC_TOOLS:
            return cls.parse_anthropic_tools(completion, validation_context, strict)

        if mode == Mode.ANTHROPIC_JSON:
            return cls.parse_anthropic_json(completion, validation_context, strict)

        if mode == Mode.VERTEXAI_TOOLS:
            return cls.parse_vertexai_tools(completion, validation_context, strict)

        if mode == Mode.VERTEXAI_JSON:
            return cls.parse_vertexai_json(completion, validation_context, strict)

        if mode == Mode.COHERE_TOOLS:
            return cls.parse_cohere_tools(completion, validation_context, strict)

        if mode == Mode.GEMINI_JSON:
            return cls.parse_gemini_json(completion, validation_context, strict)

        if mode == Mode.GEMINI_TOOLS:
            return cls.parse_gemini_tools(completion, validation_context, strict)

        if completion.choices[0].finish_reason == "length":
            raise IncompleteOutputException(last_completion=completion)

        if mode == Mode.FUNCTIONS:
            return cls.parse_functions(completion, validation_context, strict)

        if mode in {Mode.TOOLS, Mode.MISTRAL_TOOLS}:
            return cls.parse_tools(completion, validation_context, strict)

        if mode in {Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON}:
            return cls.parse_json(completion, validation_context, strict)

        raise ValueError(f"Invalid patch mode: {mode}")

    @classmethod
    def parse_anthropic_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        tool_calls = [c.input for c in completion.content if c.type == "tool_use"]  # type: ignore - TODO update with anthropic specific types

        tool_calls_validator = TypeAdapter(
            Annotated[list[Any], Field(min_length=1, max_length=1)]
        )
        tool_call = tool_calls_validator.validate_python(tool_calls)[0]

        return cls.model_validate(tool_call, context=validation_context, strict=strict)

    @classmethod
    def parse_anthropic_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from anthropic.types import Message

        assert isinstance(completion, Message)

        text = completion.content[0].text
        extra_text = extract_json_from_codeblock(text)

        if strict:
            return cls.model_validate_json(
                extra_text, context=validation_context, strict=True
            )
        else:
            # Allow control characters.
            parsed = json.loads(extra_text, strict=False)
            # Pydantic non-strict: https://docs.pydantic.dev/latest/concepts/strict_mode/
            return cls.model_validate(parsed, context=validation_context, strict=False)

    @classmethod
    def parse_gemini_json(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        try:
            text = completion.text
        except ValueError:
            logger.debug(
                f"Error response: {completion.result.candidates[0].finish_reason}\n\n{completion.result.candidates[0].safety_ratings}"
            )

        try:
            extra_text = extract_json_from_codeblock(text)  # type: ignore
        except UnboundLocalError:
            raise ValueError("Unable to extract JSON from completion text") from None

        if strict:
            return cls.model_validate_json(
                extra_text, context=validation_context, strict=True
            )
        else:
            # Allow control characters.
            parsed = json.loads(extra_text, strict=False)
            # Pydantic non-strict: https://docs.pydantic.dev/latest/concepts/strict_mode/
            return cls.model_validate(parsed, context=validation_context, strict=False)

    @classmethod
    def parse_gemini_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        tool_call = completion.parts[0].function_call
        assert (
            tool_call.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Tool name does not match"
        # reference https://ai.google.dev/gemini-api/tutorials/extract_structured_data
        response = type(tool_call).to_dict(tool_call)["args"]

        return cls.model_validate(response, context=validation_context, strict=strict)

    @classmethod
    def parse_vertexai_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        strict = False
        tool_call = completion.candidates[0].content.parts[0].function_call.args  # type: ignore
        model = {}
        for field in tool_call:  # type: ignore
            model[field] = tool_call[field]
        return cls.model_validate(model, context=validation_context, strict=strict)

    @classmethod
    def parse_vertexai_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        model = json.loads(completion.text)
        return cls.model_validate(model, context=validation_context, strict=strict)

    @classmethod
    def parse_cohere_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        text = cast(str, completion.text)  # type: ignore - TODO update with cohere specific types
        extra_text = extract_json_from_codeblock(text)
        return cls.model_validate_json(
            extra_text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_functions(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        assert (
            message.function_call.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Function name does not match"
        return cls.model_validate_json(
            message.function_call.arguments,  # type: ignore[attr-defined]
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        assert (
            len(message.tool_calls or []) == 1
        ), "Instructor does not support multiple tool calls, use List[Model] instead."
        tool_call = message.tool_calls[0]  # type: ignore
        assert (
            tool_call.function.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Tool name does not match"
        return cls.model_validate_json(
            tool_call.function.arguments,  # type: ignore
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message.content or ""
        message = extract_json_from_codeblock(message)

        return cls.model_validate_json(
            message,
            context=validation_context,
            strict=strict,
        )


def openai_schema(cls: type[BaseModel]) -> OpenAISchema:
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must be a subclass of pydantic.BaseModel")

    shema = wraps(cls, updated=())(
        create_model(
            cls.__name__ if hasattr(cls, "__name__") else str(cls),
            __base__=(cls, OpenAISchema),
        )
    )
    return cast(OpenAISchema, shema)

# --------------------------------------------------------------------------------
# The following code is adapted from a comment on GitHub in the pydantic/pydantic repository by silviumarcu.
# Source: https://github.com/pydantic/pydantic/issues/6381#issuecomment-1831607091
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
# --------------------------------------------------------------------------------

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    get_args,
    get_origin,
    Iterable,
    NoReturn,
    Optional,
    TypeVar,
)
from copy import deepcopy
import re

from instructor.mode import Mode
from instructor.dsl.partialjson import JSONParser
from instructor.utils import extract_json_from_stream, extract_json_from_stream_async

from ..anthropic_utils import xml_to_model

parser = JSONParser()
T_Model = TypeVar("T_Model", bound=BaseModel)


class PartialBase(Generic[T_Model]):
    @classmethod
    def from_streaming_response(
        cls, completion: Iterable[Any], mode: Mode, **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        if mode == Mode.ANTHROPIC_TOOLS:
            xml_chunks = cls.extract_xml(completion)
            yield from cls.model_from_xml_chunks(xml_chunks, **kwargs)
        else:
            json_chunks = cls.extract_json(completion, mode)

            if mode == Mode.MD_JSON:
                json_chunks = extract_json_from_stream(json_chunks)

            yield from cls.model_from_chunks(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls, completion: AsyncGenerator[Any, None], mode: Mode, **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        json_chunks = cls.extract_json_async(completion, mode)

        if mode == Mode.MD_JSON:
            json_chunks = extract_json_from_stream_async(json_chunks)

        return cls.model_from_chunks_async(json_chunks, **kwargs)
    
    @classmethod
    def model_from_xml_chunks(cls, xml_chunks: Iterable[Any], **kwargs: Any) -> Generator[str, None, None]:
        # note: only considers one function call
        # TODO: support self-closing tags
        # TODO: support nested structures
        accumulator = ""
        parameters_content = ""
        current_opening_tag = ""
        current_opening_tag_content = ""
        
        parameters_start_tag_regex = r"(<parameters>)"
        parameters_end_tag_regex = r"(</parameters>)"
        opening_tag_regex = r'<([a-zA-Z0-9]+)>'
        closing_tag_regex = r'</([a-zA-Z0-9]+)>'

        inside_parameters = False
        accumulating = False

        for chunk in xml_chunks:
            accumulator += chunk

            if not inside_parameters:
                start_match = re.search(parameters_start_tag_regex, accumulator, re.DOTALL)
                if start_match:
                    inside_parameters = True
                    parameters_content = ""
                    accumulator = accumulator[start_match.end():]

            while inside_parameters:
                opening_tag = re.search(opening_tag_regex, accumulator, re.DOTALL)
                closing_tag = re.search(closing_tag_regex, accumulator, re.DOTALL)
                end_match = re.search(parameters_end_tag_regex, accumulator, re.DOTALL)
                
                if end_match:
                    break

                if opening_tag and not accumulating:
                    current_opening_tag = opening_tag.group(1)
                    accumulating = True
                    accumulator = accumulator[opening_tag.end():]
                
                if closing_tag:
                    current_opening_tag_content += accumulator
                    accumulator = accumulator[closing_tag.end():]
                    if closing_tag.group(1) == current_opening_tag:
                        parameters_content += f"<{closing_tag.group(1)}>{current_opening_tag_content}"
                        current_opening_tag_content = ""
                        accumulating = False
                        try:
                            yield xml_to_model(cls, "<function_calls><invoke><tool_name></tool_name><parameters>" + parameters_content + "</parameters></invoke></function_calls>")
                        except:
                            continue

                else: # No more complete tags in the current chunk
                    break

    @classmethod
    def model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        prev_obj = None
        potential_object = ""
        for chunk in json_chunks:
            potential_object += chunk

            # Avoid parsing incomplete json when its just whitespace otherwise parser throws an exception
            task_json = (
                parser.parse(potential_object) if potential_object.strip() else None
            )
            if task_json:
                obj = cls.model_validate(task_json, strict=None, **kwargs)  # type: ignore[attr-defined]
                if obj != prev_obj:
                    obj.__dict__[
                        "chunk"
                    ] = chunk  # Provide the raw chunk for debugging and benchmarking
                    prev_obj = obj
                    yield obj

    @classmethod
    async def model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        potential_object = ""
        prev_obj = None
        async for chunk in json_chunks:
            potential_object += chunk

            # Avoid parsing incomplete json when its just whitespace otherwise parser throws an exception
            task_json = (
                parser.parse(potential_object) if potential_object.strip() else None
            )
            if task_json:
                obj = cls.model_validate(task_json, strict=None, **kwargs)  # type: ignore[attr-defined]
                if obj != prev_obj:
                    obj.__dict__[
                        "chunk"
                    ] = chunk  # Provide the raw chunk for debugging and benchmarking
                    prev_obj = obj
                    yield obj

    @staticmethod
    def extract_xml(
        completion: Iterable[Any] # not sure that this is an iterable
    ) -> Generator[str, None, None]:
        with completion as stream:
            for text in stream.text_stream:
                yield text

    @staticmethod
    def extract_json(
        completion: Iterable[Any], mode: Mode
    ) -> Generator[str, None, None]:
        for chunk in completion:
            try:
                if chunk.choices:
                    if mode == Mode.FUNCTIONS:
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif mode in {Mode.JSON, Mode.MD_JSON, Mode.JSON_SCHEMA}:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif mode == Mode.TOOLS:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            yield json_chunk[0].function.arguments
                    else:
                        raise NotImplementedError(
                            f"Mode {mode} is not supported for MultiTask streaming"
                        )
            except AttributeError:
                pass

    @staticmethod
    async def extract_json_async(
        completion: AsyncGenerator[Any, None], mode: Mode
    ) -> AsyncGenerator[str, None]:
        async for chunk in completion:
            try:
                if chunk.choices:
                    if mode == Mode.FUNCTIONS:
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif mode in {Mode.JSON, Mode.MD_JSON, Mode.JSON_SCHEMA}:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif mode == Mode.TOOLS:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            yield json_chunk[0].function.arguments
                    else:
                        raise NotImplementedError(
                            f"Mode {mode} is not supported for MultiTask streaming"
                        )
            except AttributeError:
                pass


class Partial(Generic[T_Model]):
    """Generate a new class with all attributes optionals.

    Notes:
        This will wrap a class inheriting form BaseModel and will recursively
        convert all its attributes and its children's attributes to optionals.

    Example:
        Partial[SomeModel]
    """

    def __new__(
        cls,
        *args: object,  # noqa :ARG003
        **kwargs: object,  # noqa :ARG003
    ) -> "Partial[T_Model]":
        """Cannot instantiate.

        Raises:
            TypeError: Direct instantiation not allowed.
        """
        raise TypeError("Cannot instantiate abstract Partial class.")

    def __init_subclass__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> NoReturn:
        """Cannot subclass.

        Raises:
           TypeError: Subclassing not allowed.
        """
        raise TypeError("Cannot subclass {}.Partial".format(cls.__module__))

    def __class_getitem__(  # type: ignore[override]
        cls,
        wrapped_class: type[T_Model],
    ) -> type[T_Model]:
        """Convert model to a partial model with all fields being optionals."""

        def _make_field_optional(
            field: FieldInfo,
        ) -> tuple[object, FieldInfo]:
            tmp_field = deepcopy(field)

            annotation = field.annotation

            # Handle generics (like List, Dict, etc.)
            if get_origin(annotation) is not None:
                # Get the generic base (like List, Dict) and its arguments (like User in List[User])
                generic_base = get_origin(annotation)
                generic_args = get_args(annotation)

                # Recursively apply Partial to each of the generic arguments
                modified_args = tuple(
                    Partial[arg]  # type: ignore[valid-type]
                    if isinstance(arg, type) and issubclass(arg, BaseModel)
                    else arg
                    for arg in generic_args
                )

                # Reconstruct the generic type with modified arguments
                tmp_field.annotation = (
                    Optional[generic_base[modified_args]] if generic_base else None
                )
                tmp_field.default = None
            # If the field is a BaseModel, then recursively convert it's
            # attributes to optionals.
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                tmp_field.annotation = Optional[Partial[annotation]]  # type: ignore[assignment, valid-type]
                tmp_field.default = {}
            else:
                tmp_field.annotation = Optional[field.annotation]  # type: ignore[assignment]
                tmp_field.default = None
            return tmp_field.annotation, tmp_field

        return create_model(
            __model_name=f"Partial{wrapped_class.__name__}",
            __base__=(wrapped_class, PartialBase),
            __module__=wrapped_class.__module__,
            **{
                field_name: _make_field_optional(field_info)
                for field_name, field_info in wrapped_class.__fields__.items()
            },
        )  # type: ignore[all]

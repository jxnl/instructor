from enum import Enum
from typing import List
from typing_extensions import Literal

import anthropic
import pytest
from pydantic import BaseModel, Field

import instructor
from instructor.anthropic_utils import build_xml_from_schema

create = instructor.patch(
    create=anthropic.Anthropic().messages.create, mode=instructor.Mode.ANTHROPIC_TOOLS
)


@pytest.mark.skip
def test_anthropic():
    class Properties(BaseModel):
        name: str
        value: List[str]

    class User(BaseModel):
        name: str
        age: int
        properties: List[Properties]

    resp = create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and properties.",
            }
        ],
        response_model=User,
    )  # type: ignore

    assert isinstance(resp, User)


@pytest.mark.skip
def test_anthropic_enum():
    class ProgrammingLanguage(Enum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        UNKNOWN = "unknown"
        OTHER = "other"

    class SimpleEnum(BaseModel):
        language: ProgrammingLanguage

    resp = create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "What is your favorite programming language?",
            }
        ],
        response_model=SimpleEnum,
    )  # type: ignore

    assert isinstance(resp, SimpleEnum)


def test_simple_model():
    class SimpleModel(BaseModel):
        a: int = Field(..., description="idc")
        b: int = 2

    json_schema = SimpleModel.model_json_schema()
    output_xml = build_xml_from_schema(json_schema)

    expected = "<tool_description><tool_name>SimpleModel</tool_name><properties><property><name>a</name><type>integer</type></property><property><name>b</name><type>integer</type></property></properties></tool_description>"
    assert output_xml == expected


def test_simple_nested_model():
    class Number(BaseModel):
        number: int

    class NestedModel(BaseModel):
        nested_model: Number = Field(..., description="idc")

    json_schema = NestedModel.model_json_schema()
    output_xml = build_xml_from_schema(json_schema)

    expected = "<tool_description><tool_name>NestedModel</tool_name><properties><property><name>nested_model</name><type>object</type><properties><property><name>number</name><type>integer</type></property></properties></property></properties></tool_description>"
    assert output_xml == expected

def test_simple_list_model():
    class ListSimple(BaseModel):
        list_of_ints: List[int]

    json_schema = ListSimple.model_json_schema()
    output_xml = build_xml_from_schema(json_schema)

    expected = "<tool_description><tool_name>ListSimple</tool_name><properties><property><name>list_of_ints</name><type>List[integer]</type></property></properties></tool_description>"
    assert output_xml == expected


def test_complex_list_model():
    class Properties(BaseModel):
        name: str
        value: List[str]


    class ListObject(BaseModel):
        name: str
        age: int
        properties: List[Properties]

    json_schema = ListObject.model_json_schema()
    output_xml = build_xml_from_schema(json_schema)

    expected = "<tool_description><tool_name>ListObject</tool_name><properties><property><name>name</name><type>string</type></property><property><name>age</name><type>integer</type></property><property><name>properties</name><type>List</type><items><properties><property><name>name</name><type>string</type></property><property><name>value</name><type>List[string]</type></property></properties></items></property></properties></tool_description>"
    assert output_xml == expected

def test_simple_literal():
    class SimpleLiteral(BaseModel):
        """Testing"""
        languages: Literal["python", "javascript", "typescript", "java", "haskell", "php"]

    json_schema = SimpleLiteral.model_json_schema()
    output_xml = build_xml_from_schema(json_schema)

    expected = "<tool_description><tool_name>SimpleLiteral</tool_name><description>Testing</description><properties><property><name>languages</name><type>string</type><values><value>python</value><value>javascript</value><value>typescript</value><value>java</value><value>haskell</value><value>php</value></values></property></properties></tool_description>"
    assert output_xml == expected

def test_simple_enum():
    class ProgrammingLanguage(Enum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        UNKNOWN = "unknown"
        OTHER = "other"


    class SimpleEnum(BaseModel):
        language: ProgrammingLanguage

    json_schema = SimpleEnum.model_json_schema()
    output_xml = build_xml_from_schema(json_schema)

    expected = "<tool_description><tool_name>SimpleEnum</tool_name><properties><property><name>language</name><type>string</type><values><value>python</value><value>javascript</value><value>typescript</value><value>unknown</value><value>other</value></values></property></properties></tool_description>"
    assert output_xml == expected
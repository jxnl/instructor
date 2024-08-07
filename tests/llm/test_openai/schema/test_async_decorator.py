from instructor import async_field_validator, AsyncInstructMixin
from instructor.function_calls import openai_schema
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Union, Literal


class UserExtractValidated(BaseModel, AsyncInstructMixin):
    name: str
    age: int

    @async_field_validator("name")
    async def validate_name(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError(
                f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
            )
        return v


class User(BaseModel, AsyncInstructMixin):
    name: str = Field(description="User's name")
    age: int = Field(description="User's age")
    email: str = Field(description="User's email address")


def test_openai_schema_serialization():
    UserSchema = openai_schema(User)
    assert User.model_json_schema() == UserSchema.model_json_schema()


def test_openai_schema_float_and_bool():
    class FloatBoolModel(BaseModel, AsyncInstructMixin):
        price: float = Field(description="Price of an item")
        is_available: bool = Field(description="Availability status")

    FloatBoolSchema = openai_schema(FloatBoolModel)
    assert FloatBoolModel.model_json_schema() == FloatBoolSchema.model_json_schema()


def test_openai_schema_bytes_and_literal():
    class BytesLiteralModel(BaseModel, AsyncInstructMixin):
        data: bytes = Field(description="Binary data")
        status: Literal["active", "inactive"] = Field(description="Current status")

    BytesLiteralSchema = openai_schema(BytesLiteralModel)
    assert (
        BytesLiteralModel.model_json_schema() == BytesLiteralSchema.model_json_schema()
    )


def test_nested_class():
    class Users(BaseModel, AsyncInstructMixin):
        users: list[User]
        user: User

    assert Users.model_json_schema() == openai_schema(Users).model_json_schema()


def test_nested_class_with_async_decorators():
    class NestedUserWithValidation(BaseModel, AsyncInstructMixin):
        name: str

        @async_field_validator("name")
        async def validate_uppercase(self, v: str) -> str:
            await asyncio.sleep(2)
            return v

    class Users(BaseModel):
        users: list[NestedUserWithValidation]

    assert Users.model_json_schema() == openai_schema(Users).model_json_schema()


def test_schema_optional_and_enum():
    class QueryType(str, Enum):
        DOCUMENT_CONTENT = "document_content"
        LAST_MODIFIED = "last_modified"
        ACCESS_PERMISSIONS = "access_permissions"
        RELATED_DOCUMENTS = "related_documents"

    # Define the structure for query responses
    class QueryResponse(BaseModel, AsyncInstructMixin):
        query_type: QueryType
        response: str
        additional_info: Optional[str] = None

    assert (
        openai_schema(QueryResponse).model_json_schema()
        == QueryResponse.model_json_schema()
    )


def test_schema_union():
    class Search(BaseModel, AsyncInstructMixin):
        search_query: str

    class CalendarInvite(BaseModel, AsyncInstructMixin):
        date: str

    # Define the structure for query responses
    class QueryResponse(BaseModel, AsyncInstructMixin):
        query_type: Union[Search, CalendarInvite]
        response: str
        additional_info: Optional[str] = None

    assert (
        openai_schema(QueryResponse).model_json_schema()
        == QueryResponse.model_json_schema()
    )

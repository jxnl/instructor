from typing import TypeVar


from datetime import datetime, date, time
from pydantic import BaseModel
from instructor import (
    openai_schema,
    AsyncInstructMixin,
    async_field_validator,
    async_model_validator,
)
from decimal import Decimal
from uuid import UUID
from typing import Annotated, Union, Optional, Literal, Any
from collections import OrderedDict
from enum import Enum
import asyncio
from pydantic import BaseModel, Field

T = TypeVar("T")


def test_annotation_schema():
    class User(BaseModel):
        details: dict[
            Annotated[str, Field(description="User name", min_length=1)],
            Annotated[int, Field(description="User ID", gt=3)],
        ] = Field(max_length=1)

    original_schema = openai_schema(User).model_json_schema()

    User.__bases__ += (AsyncInstructMixin,)
    assert issubclass(User, AsyncInstructMixin)
    assert original_schema == openai_schema(User).model_json_schema()


class User(BaseModel):
    name: str
    age: int


class AdminUser(BaseModel):
    organization: str
    name: str
    email: str


def test_new_union_types():
    import sys

    if sys.version_info >= (3, 10):

        class Users(BaseModel):
            users: list[AdminUser | User]

        original_schema = openai_schema(Users).model_json_schema()

        Users.__bases__ += (AsyncInstructMixin,)
        assert issubclass(Users, AsyncInstructMixin)
        assert openai_schema(Users).model_json_schema() == original_schema


def test_old_union_type():
    class UsersOldUnion(BaseModel):
        users: list[Union[AdminUser, User]]

    original_schema = openai_schema(UsersOldUnion).model_json_schema()

    UsersOldUnion.__bases__ += (AsyncInstructMixin,)
    assert issubclass(UsersOldUnion, AsyncInstructMixin)
    assert openai_schema(UsersOldUnion).model_json_schema() == original_schema


def test_tuple_with_multiple_args():
    class TupleModel(BaseModel):
        coordinates: tuple[int, int, int]
        name_and_age: tuple[str, int]

    original_schema = openai_schema(TupleModel).model_json_schema()

    TupleModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(TupleModel, AsyncInstructMixin)
    assert openai_schema(TupleModel).model_json_schema() == original_schema


def test_dict_with_multiple_value_types():
    from collections import OrderedDict

    class DictModel(BaseModel):
        regular_dict: dict[str, Union[int, str]]
        ordered_dict: OrderedDict[str, Union[float, bool]]

    original_schema = openai_schema(DictModel).model_json_schema()

    DictModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(DictModel, AsyncInstructMixin)
    assert openai_schema(DictModel).model_json_schema() == original_schema


def test_nested_complex_types():
    class ComplexModel(BaseModel):
        nested_tuple_dict: dict[str, tuple[int, str, bool]]
        list_of_dicts: list[dict[str, Union[int, str]]]

    original_schema = openai_schema(ComplexModel).model_json_schema()

    ComplexModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(ComplexModel, AsyncInstructMixin)
    assert openai_schema(ComplexModel).model_json_schema() == original_schema


def test_openai_schema_tuple_mapping():
    class TestModel(BaseModel):
        field: tuple[str, int, int]

    original_schema = openai_schema(TestModel).model_json_schema()

    TestModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(TestModel, AsyncInstructMixin)
    assert openai_schema(TestModel).model_json_schema() == original_schema


def test_openai_schema_dict_mapping():
    class TestModel(BaseModel):
        field: dict[str, str]

    original_schema = openai_schema(TestModel).model_json_schema()

    TestModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(TestModel, AsyncInstructMixin)
    assert openai_schema(TestModel).model_json_schema() == original_schema


def test_openai_schema_ordered_dict_mapping():
    class TestModel(BaseModel):
        field: OrderedDict[str, int]

    original_schema = openai_schema(TestModel).model_json_schema()

    TestModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(TestModel, AsyncInstructMixin)
    assert openai_schema(TestModel).model_json_schema() == original_schema


def test_openai_schema_supports_optional_none_310():
    class DummyWithOptionalNone(BaseModel):
        """
        Class with a single attribute that can be a string or None.
        Validates support of UnionType in schema generation.
        """

        attr: str | None

    original_schema = openai_schema(DummyWithOptionalNone).model_json_schema()

    DummyWithOptionalNone.__bases__ += (AsyncInstructMixin,)
    assert issubclass(DummyWithOptionalNone, AsyncInstructMixin)
    assert openai_schema(DummyWithOptionalNone).model_json_schema() == original_schema


def test_openai_schema_supports_optional_none() -> None:
    class DummyWithOptionalNone(BaseModel):
        """
        Class with a single attribute that can be a string or None.
        Validates support of UnionType in schema generation.
        """

        attr: Optional[str]  # In python 3.10+ this is written as `attr: str | None`
        attr2: Union[str, None]

    original_schema = openai_schema(DummyWithOptionalNone).model_json_schema()

    DummyWithOptionalNone.__bases__ += (AsyncInstructMixin,)
    assert issubclass(DummyWithOptionalNone, AsyncInstructMixin)
    assert openai_schema(DummyWithOptionalNone).model_json_schema() == original_schema


def test_default_values_and_validators():
    class UserWithDefaults(BaseModel):
        name: str = "John Doe"
        age: int = Field(default=30, ge=0)

    original_schema = openai_schema(UserWithDefaults).model_json_schema()

    UserWithDefaults.__bases__ += (AsyncInstructMixin,)
    assert issubclass(UserWithDefaults, AsyncInstructMixin)
    assert openai_schema(UserWithDefaults).model_json_schema() == original_schema


def test_inheritance():
    class BaseUser(BaseModel):
        name: str

    class ExtendedUser(BaseUser):
        age: int

    original_schema = openai_schema(ExtendedUser).model_json_schema()

    BaseUser.__bases__ += (AsyncInstructMixin,)
    ExtendedUser.__bases__ += (AsyncInstructMixin,)
    assert issubclass(BaseUser, AsyncInstructMixin)
    assert issubclass(ExtendedUser, AsyncInstructMixin)
    assert openai_schema(ExtendedUser).model_json_schema() == original_schema


def test_alias_and_field_customization():
    class AliasModel(BaseModel):
        actual_name: str = Field(..., alias="name")
        age: int = Field(..., title="User Age", description="The age of the user")

    original_schema = openai_schema(AliasModel).model_json_schema()

    AliasModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(AliasModel, AsyncInstructMixin)
    assert openai_schema(AliasModel).model_json_schema() == original_schema


def test_standard_python_types():
    class StandardTypesModel(BaseModel):
        timestamp: datetime
        date_field: date
        time_field: time
        price: Decimal
        unique_id: UUID

    original_schema = openai_schema(StandardTypesModel).model_json_schema()

    StandardTypesModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(StandardTypesModel, AsyncInstructMixin)
    assert openai_schema(StandardTypesModel).model_json_schema() == original_schema


def test_any_type():
    class AnyTypeModel(BaseModel):
        any_field: Any

    original_schema = openai_schema(AnyTypeModel).model_json_schema()

    AnyTypeModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(AnyTypeModel, AsyncInstructMixin)
    assert openai_schema(AnyTypeModel).model_json_schema() == original_schema


def test_literal_type():
    class LiteralTypeModel(BaseModel):
        status: Literal["active", "inactive", "pending"]

    original_schema = openai_schema(LiteralTypeModel).model_json_schema()

    class LiteralTypeModel(BaseModel, AsyncInstructMixin):
        status: Literal["active", "inactive", "pending"]

    assert openai_schema(LiteralTypeModel).model_json_schema() == original_schema


def test_str_any_dict():
    import sys

    if sys.version_info >= (3, 10):

        class ChatResponse(BaseModel):
            action_data: dict[str, Any] | None = Field(
                default=None,
                description="The required data for the action that will be performed.",
            )

            content: str = Field(
                description="A contextual response to the user's message."
            )
    else:

        class ChatResponse(BaseModel):
            action_data: Union[dict[str, Any], None] = Field(
                default=None,
                description="The required data for the action that will be performed.",
            )

    original_schema = openai_schema(ChatResponse).model_json_schema()

    if sys.version_info >= (3, 10):

        class ChatResponse(BaseModel, AsyncInstructMixin):
            action_data: dict[str, Any] | None = Field(
                default=None,
                description="The required data for the action that will be performed.",
            )

            content: str = Field(
                description="A contextual response to the user's message."
            )
    else:

        class ChatResponse(BaseModel, AsyncInstructMixin):
            action_data: Union[dict[str, Any], None] = Field(
                default=None,
                description="The required data for the action that will be performed.",
            )

    assert original_schema == openai_schema(ChatResponse).model_json_schema()


def test_openai_schema_parser_with_field_validator():
    class AdminUser(BaseModel, AsyncInstructMixin):
        name: str
        age: int
        email: str

        @async_field_validator("name")
        def validate_name_admin(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class User(BaseModel, AsyncInstructMixin):
        name: str
        age: int

        @async_field_validator("name")
        def validate_name_user(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class Users(BaseModel, AsyncInstructMixin):
        users: list[Union[User, AdminUser]]

    mixin_schema = openai_schema(Users).model_json_schema()

    class AdminUser(BaseModel):
        name: str
        age: int
        email: str

        @async_field_validator("name")
        def validate_name_admin(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class User(BaseModel):
        name: str
        age: int

        @async_field_validator("name")
        def validate_name_user(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class Users(BaseModel):
        users: list[Union[User, AdminUser]]

    original_schema = openai_schema(Users).model_json_schema()
    assert original_schema == mixin_schema


def test_openai_schema_float_and_bool():
    class FloatBoolModel(BaseModel):
        price: float = Field(description="Price of an item")
        is_available: bool = Field(description="Availability status")

    original_schema = openai_schema(FloatBoolModel).model_json_schema()

    FloatBoolModel.__bases__ += (AsyncInstructMixin,)
    assert issubclass(FloatBoolModel, AsyncInstructMixin)
    assert openai_schema(FloatBoolModel).model_json_schema() == original_schema


def test_openai_schema_bytes_and_literal():
    class BytesLiteralModel(BaseModel):
        data: bytes = Field(description="Binary data")
        status: Literal["active", "inactive"] = Field(description="Current status")

    original_schema = openai_schema(BytesLiteralModel).model_json_schema()

    BytesLiteralModel.__bases__ += (AsyncInstructMixin,)

    assert issubclass(BytesLiteralModel, AsyncInstructMixin)
    assert openai_schema(BytesLiteralModel).model_json_schema() == original_schema


def test_schema_union():
    class Search(BaseModel):
        search_query: str

    class CalendarInvite(BaseModel):
        date: str

    class QueryResponse(BaseModel):
        query_type: Union[Search, CalendarInvite]
        response: str
        additional_info: Optional[str] = None

    original_schema = openai_schema(QueryResponse).model_json_schema()

    # Add AsyncInstructMixin to the existing classes
    Search.__bases__ += (AsyncInstructMixin,)
    CalendarInvite.__bases__ += (AsyncInstructMixin,)
    QueryResponse.__bases__ += (AsyncInstructMixin,)

    assert issubclass(QueryResponse, AsyncInstructMixin)
    assert openai_schema(QueryResponse).model_json_schema() == original_schema


def test_schema_optional_and_enum():
    class QueryType(str, Enum):
        DOCUMENT_CONTENT = "document_content"
        LAST_MODIFIED = "last_modified"
        ACCESS_PERMISSIONS = "access_permissions"
        RELATED_DOCUMENTS = "related_documents"

    # Define the structure for query responses
    class QueryResponse(BaseModel):
        query_type: QueryType
        response: str
        additional_info: Optional[str] = None

    original_schema = openai_schema(QueryResponse).model_json_schema()

    QueryResponse.__bases__ += (AsyncInstructMixin,)
    assert issubclass(QueryResponse, AsyncInstructMixin)
    assert openai_schema(QueryResponse).model_json_schema() == original_schema


def test_nested_class_with_async_field_decorators():
    class NestedUserWithValidation(BaseModel):
        name: str

        @async_field_validator("name")
        async def validate_uppercase(self, v: str) -> str:
            await asyncio.sleep(2)
            return v

    class Users(BaseModel):
        users: list[NestedUserWithValidation]

    original_schema = openai_schema(Users).model_json_schema()

    Users.__bases__ += (AsyncInstructMixin,)
    assert issubclass(Users, AsyncInstructMixin)
    assert openai_schema(Users).model_json_schema() == original_schema


def test_nested_class_with_async_model_decorators():
    class NestedUserWithValidation(BaseModel):
        name: str

        @async_model_validator()
        async def validate_uppercase(self):
            await asyncio.sleep(2)
            if not self.name.isupper():
                raise ValueError("Name must be uppercase")

    class Users(BaseModel):
        users: list[NestedUserWithValidation]

    original_schema = openai_schema(Users).model_json_schema()

    Users.__bases__ += (AsyncInstructMixin,)
    assert issubclass(Users, AsyncInstructMixin)
    assert openai_schema(Users).model_json_schema() == original_schema

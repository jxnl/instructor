from pydantic import BaseModel, ValidationInfo
from instructor import async_field_validator, AsyncInstructMixin, async_model_validator
import pytest
import asyncio
from typing import Union


@pytest.mark.asyncio
async def test_field_validator():
    class User(BaseModel, AsyncInstructMixin):
        name: str
        label: str

        @async_field_validator("name", "label")
        async def validate_user(self, v: str):
            await asyncio.sleep(3)
            if not v.isupper():
                raise ValueError(f"Uppercase response required for {v}")

    exceptions = await User(**{"name": "tom", "label": "active"}).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required for tom encountered at name",
        "Exception of Uppercase response required for active encountered at label",
    ]


@pytest.mark.asyncio
async def test_multiple_field_validators():
    class User(BaseModel, AsyncInstructMixin):
        name: str
        label: str

        @async_field_validator("name", "label")
        def validate_user(self, v: str):
            if not v.isupper():
                raise ValueError(f"Uppercase response required for {v}")

    exceptions = await User(**{"name": "tom", "label": "active"}).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required for tom encountered at name",
        "Exception of Uppercase response required for active encountered at label",
    ]


@pytest.mark.asyncio
async def test_model_validator():
    class User(BaseModel, AsyncInstructMixin):
        name: str
        label: str

        @async_model_validator()
        def validate_user(self):
            if not self.name.isupper() or not self.label.isupper():
                raise ValueError(f"Uppercase response required")

    exceptions = await User(**{"name": "tom", "label": "active"}).model_async_validate()

    assert len(exceptions) == 1
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required encountered",
    ]


@pytest.mark.asyncio
async def test_parsing_nested_field():
    class UserExtractValidated(BaseModel, AsyncInstructMixin):
        name: str
        age: int

        @async_field_validator("name")
        async def validate_name(cls, v: str) -> str:
            await asyncio.sleep(3)
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class Users(BaseModel, AsyncInstructMixin):
        users: list[UserExtractValidated]

    exceptions = await Users(
        **{"users": [{"name": "thomas", "age": 27}, {"name": "vincent", "age": 24}]}
    ).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of All Letters in the name must be uppercased. thomas is not a valid response. Eg JASON, TOM not jason, tom encountered at users.name",
        "Exception of All Letters in the name must be uppercased. vincent is not a valid response. Eg JASON, TOM not jason, tom encountered at users.name",
    ]


@pytest.mark.asyncio
async def test_context_passing_in_nested_model():
    class ModelValidationCheck(BaseModel, AsyncInstructMixin):
        user_names: list[str]

        @async_model_validator()
        async def validate_model(self, info: ValidationInfo):
            await asyncio.sleep(3)
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(BaseModel, AsyncInstructMixin):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        **{"model": {"user_names": ["Jack", "Thomas", "Ben"]}}
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert (
        str(res[0])
        == "Exception of Invalid Error but with {'abcdef': '123'}! encountered at model"
    )


@pytest.mark.asyncio
async def test_context_passing_in_nested_field_validator():
    class ModelValidationCheck(BaseModel, AsyncInstructMixin):
        user_names: list[str]

        @async_field_validator("user_names")
        async def validate_model(self, v: list[str], info: ValidationInfo):
            await asyncio.sleep(3)
            assert len(v) > 0
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(BaseModel, AsyncInstructMixin):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        **{"model": {"user_names": ["Jack", "Thomas", "Ben"]}}
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert (
        str(res[0])
        == "Exception of Invalid Error but with {'abcdef': '123'}! encountered at model.user_names"
    )


@pytest.mark.asyncio
async def test_nested_union_validator():
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

    resp = Users(
        **{
            "users": [
                {
                    "name": "thomas",
                    "age": 27,
                },
                {"name": "vincent", "age": 24, "email": "vincent@gmail.com"},
                {"name": "johann", "age": 36, "email": "vincent@gmail.com"},
            ]
        }
    )

    assert len(await resp.model_async_validate()) == 3


def test_nested_class_with_async_field_decorators():
    class UserWithAsyncValidators(BaseModel, AsyncInstructMixin):
        name: str
        age: int

        @async_field_validator("name")
        async def validate_name(cls, v: str) -> str:
            await asyncio.sleep(0.1)
            return v.strip()

        @async_model_validator()
        async def validate_age(self) -> "UserWithAsyncValidators":
            await asyncio.sleep(0.1)
            if self.age < 0:
                raise ValueError("Age must be non-negative")
            return self

    class UserWithoutAsyncValidators(BaseModel, AsyncInstructMixin):
        name: str
        age: int

    user_with = UserWithAsyncValidators(name="John", age=30)
    user_without = UserWithoutAsyncValidators(name="Jane", age=25)

    assert user_with.has_async_validators() is True
    assert user_without.has_async_validators() is False

    class NestedUsers(BaseModel, AsyncInstructMixin):
        user_with: UserWithAsyncValidators
        user_without: UserWithoutAsyncValidators

    nested_users = NestedUsers(
        user_with={"name": "John", "age": 30},
        user_without={"name": "Jane", "age": 25},
    )

    assert nested_users.has_async_validators() is True

    class AllWithoutValidators(BaseModel, AsyncInstructMixin):
        users: list[UserWithoutAsyncValidators]

    all_without = AllWithoutValidators(
        **{
            "users": [
                {"name": "Jane", "age": 25},
                {"name": "Jane", "age": 25},
            ]
        }
    )
    assert all_without.has_async_validators() == False

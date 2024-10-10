import instructor
from fireworks.client import Fireworks, AsyncFireworks
from pydantic import BaseModel, field_validator
from collections.abc import Iterable
import pytest

modes = [
    (instructor.Mode.FIREWORKS_JSON, "accounts/fireworks/models/llama-v3-70b-instruct"),
    (instructor.Mode.FIREWORKS_TOOLS, "accounts/fireworks/models/firefunction-v2"),
]


@pytest.mark.parametrize("mode, model", modes)
def test_fireworks_tools(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(Fireworks(), mode=mode)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("mode, model", modes)
def test_fireworks_tools_validated(mode: instructor.Mode, model: str):
    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_validator(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All letters in the name must be uppercase (Eg. JOHN, SMITH) - {v} is not a valid example."
                )
            return v

    client = instructor.from_fireworks(Fireworks(), mode=mode)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        max_retries=5,
        response_model=ValidatedUser,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


@pytest.mark.parametrize("mode, model", modes)
@pytest.mark.asyncio(scope="session")
async def test_async_fireworks(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(AsyncFireworks(), mode=mode)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("mode, model", modes)
@pytest.mark.asyncio(scope="session")
async def test_async_fireworks_retries(mode: instructor.Mode, model: str):
    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_validator(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"Make sure to uppercase all letters in the name field. Examples include: JOHN, SMITH, etc. {v} is not a valid example of a name that has all its letters uppercased"
                )
            return v

    client = instructor.from_fireworks(AsyncFireworks(), mode=mode)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=ValidatedUser,
        max_retries=5,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


def test_fireworks_json_streaming():
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(Fireworks(), mode=instructor.Mode.FIREWORKS_JSON)

    resp = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Extract all users from this sentence : Ivan is 27 and lives in Singapore. Darren is around the same age and lives in the same city. Make sure to adhere to the desired JSON format.",
            },
        ],
        response_model=Iterable[User],
        stream=True,
    )

    users = [user for user in resp]

    assert len(users) == 2
    assert {user.name for user in users} == {"Ivan", "Darren"}
    assert {user.age for user in users} == {27}


@pytest.mark.asyncio
async def test_fireworks_json_async_streaming():
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(
        AsyncFireworks(), mode=instructor.Mode.FIREWORKS_JSON
    )

    resp = await client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Extract all users from this sentence : Ivan is 27 and lives in Singapore. Darren is around the same age and lives in the same city",
            },
        ],
        response_model=Iterable[User],
        stream=True,
        max_retries=5,
    )

    async for user in resp:
        assert isinstance(user, User)


def test_fireworks_tool_error():
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(Fireworks(), mode=instructor.Mode.FIREWORKS_TOOLS)

    with pytest.raises(ValueError):
        resp = client.chat.completions.create(
            model="accounts/fireworks/models/firefunction-v2",
            messages=[
                {
                    "role": "user",
                    "content": "Extract all users from this sentence : Ivan is 27 and lives in Singapore. Darren is around the same age and lives in the same city",
                },
            ],
            response_model=Iterable[User],
            stream=True,
        )

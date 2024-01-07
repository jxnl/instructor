from itertools import product
from typing import Iterable
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError, validator

import instructor
from tests.openai.util import models, modes


class User(BaseModel):
    name: str
    age: int


Users = Iterable[User]


class NotBob(BaseModel):
    name: str

    @validator("name")
    def name_cannot_be_alice(cls, value):
        if value.lower() == "bob":
            raise ValueError('name cannot be "bob"')
        return value



@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multi_user(model, mode, client):
    client = instructor.patch(client, mode=mode)
    def stream_extract(input: str) -> Iterable[User]:
        return client.chat.completions.create(
            model=model,
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30")]
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, modes))
async def test_multi_user_tools_mode_async(model, mode, aclient):
    client = instructor.patch(aclient, mode=mode)

    async def stream_extract(input: str) -> Iterable[User]:
        return await client.chat.completions.create(
            model=model,
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = []
    async for user in await stream_extract(input="Jason is 20, Sarah is 30"):
        resp.append(user)
    print(resp)
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS, Mode.MD_JSON])
def test_stream_objects_with_throw_stream_exceptions(mode):
    client = instructor.patch(OpenAI(), mode=mode)
    with pytest.raises(ValidationError):
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            stream=True,
            response_model=Iterable[NotBob],
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {
                    "role": "user",
                    "content": "Return a list of 3 JSON objects. The first object must have a key 'name' with the value 'alice'. The second object must have a key 'name' with the value 'bob'. The third object must have a key 'name' with the value 'carol'.",
                },
            ],
            max_tokens=1000,
            throw_stream_exceptions=True,
        )
        for user in stream:
            assert isinstance(user, NotBob)


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS, Mode.MD_JSON])
def test_stream_objects_without_throw_stream_exceptions(mode):
    client = instructor.patch(OpenAI(), mode=mode)

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        stream=True,
        response_model=Iterable[NotBob],
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {
                "role": "user",
                "content": "Return a list of 3 JSON objects. The first object must have a key 'name' with the value 'alice'. The second object must have a key 'name' with the value 'bob'. The third object must have a key 'name' with the value 'carol'.",
            },
        ],
        max_tokens=1000,
        throw_stream_exceptions=False,
    )

    users = [user for user in stream]

    assert len(users) == 2
    assert users[0].name == "alice"
    assert users[1].name == "carol"

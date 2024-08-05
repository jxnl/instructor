import pytest
from pydantic import BaseModel, Field, field_validator
import instructor
from instructor.mode import Mode


modes = [Mode.COHERE_JSON_SCHEMA, Mode.COHERE_TOOLS]


class User(BaseModel):
    name: str = Field("User's first name")
    age: int


@pytest.mark.parametrize("mode", modes)
def test_parse_user_sync(client, mode):
    client = instructor.from_cohere(client, mode=mode)

    resp = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
            }
        ],
    )

    assert resp.name == "Ivan"
    assert resp.age == 27


class ValidatedUser(BaseModel):
    name: str = Field("User's first name")
    age: int

    @field_validator("name")
    @classmethod
    def ensure_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError(
                f"{v} should have all its characters uppercased (Eg. TOM, JACK, JEFFREY)"
            )
        return v


@pytest.mark.parametrize("mode", modes)
def test_parse_validated_user_sync(client, mode):
    client = instructor.from_cohere(client, mode=mode)

    resp = client.chat.completions.create(
        response_model=ValidatedUser,
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
            }
        ],
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


@pytest.mark.parametrize("mode", modes)
@pytest.mark.asyncio
async def test_parse_user_async(aclient, mode):
    client = instructor.from_cohere(aclient, mode=mode)

    resp = await client.chat.completions.create(
        response_model=ValidatedUser,
        model="command-r-plus",
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
            }
        ],
        max_retries=4,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27

from itertools import product
from pydantic import BaseModel, field_validator
import pytest
import instructor
import google.generativeai as genai

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_runmodel(model, mode):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)
    model = client.chat.completions.create(
        response_model=UserExtract,
        max_retries=2,
        strict=False,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert model.age == 25
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from Gemini"


class UserExtractValidated(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError(
                "Name should be uppercase, make sure to use the `uppercase` version of the name"
            )
        return v


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_runmodel_validator(model, mode):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)
    model = client.chat.completions.create(
        response_model=UserExtractValidated,
        strict=False,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from Gemini"

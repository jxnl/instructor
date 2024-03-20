import anthropic
import instructor
from pydantic import BaseModel
from typing import List

create = instructor.patch(
    create=anthropic.Anthropic().messages.create,
    mode=instructor.Mode.ANTHROPIC_TOOLS)

def test_simple():
    class User(BaseModel):
        name: str
        age: int
    
    resp = create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old.",
            }
        ],
        response_model=User,
    ) # type: ignore

    assert isinstance(resp, User)
    assert resp.name == "John"
    assert resp.age == 18

def test_nested_type():
    class Address(BaseModel):
        house_number: int
        street_name: str

    class User(BaseModel):
        name: str
        age: int
        address: Address
    
    resp = create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old and lives at 123 First Avenue.",
            }
        ],
        response_model=User,
    ) # type: ignore

    assert isinstance(resp, User)
    assert resp.name == "John"
    assert resp.age == 18
    
    assert isinstance(resp.address, Address)
    assert resp.address.house_number == 123
    assert resp.address.street_name == "First Avenue"

def test_nested_list():
    class Properties(BaseModel):
        key: str
        value: str

    class User(BaseModel):
        name: str
        age: int
        properties: List[Properties]

    resp = create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and properties.",
            }
        ],
        response_model=User,
    ) # type: ignore

    assert isinstance(resp, User)
    for property in resp.properties:
        assert isinstance(property, Properties)
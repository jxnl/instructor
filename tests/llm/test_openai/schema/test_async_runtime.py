from pydantic import BaseModel
from instructor import async_field_validator, AsyncInstructMixin
import pytest
import asyncio


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

    class Users(BaseModel, AsyncInstructMixin):
        users: list[User]

    import time

    fake_users = [
        {"name": "alice", "label": "user"},
        {"name": "bob", "label": "admin"},
        {"name": "charlie", "label": "moderator"},
        {"name": "david", "label": "guest"},
        {"name": "eve", "label": "user"},
    ]

    start_time = time.time()
    exceptions = await Users(**{"users": fake_users}).model_async_validate()
    end_time = time.time()

    total_time = end_time - start_time
    assert (
        total_time < 6
    ), f"Validation took {total_time:.2f} seconds, which is more than 6 seconds"

    assert len(exceptions) == 10  # 2 exceptions per user (name and label), 5 users
    expected_exceptions = [
        f"Exception of Uppercase response required for {user['name']} encountered at users.name"
        for user in fake_users
    ] + [
        f"Exception of Uppercase response required for {user['label']} encountered at users.label"
        for user in fake_users
    ]
    assert set([str(item) for item in exceptions]) == set(expected_exceptions)

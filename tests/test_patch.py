import functools

from openai import AsyncOpenAI, OpenAI

import instructor
from instructor.utils import OVERRIDE_DOCS, is_async


def test_patch_completes_successfully():
    instructor.patch(OpenAI())


def test_apatch_completes_successfully():
    instructor.apatch(AsyncOpenAI())


def test_is_async_returns_true_if_function_is_async():
    async def async_function():
        pass

    assert is_async(async_function) is True


def test_is_async_returns_false_if_function_is_not_async():
    def sync_function():
        pass

    assert is_async(sync_function) is False


def test_is_async_returns_true_if_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    assert is_async(wrapped_function) is True


def test_override_docs():
    assert (
        "response_model" in OVERRIDE_DOCS
    ), "response_model should be in OVERRIDE_DOCS"

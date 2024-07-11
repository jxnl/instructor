# Usage Tracking

Instructor provides two methods to track token usage for your API calls: the new recommended method using `with_usage=True`, and the previous method of accessing the raw response. The new method offers more comprehensive coverage across different providers (OpenAI, Anthropic, Gemini, etc.).

## Recommended Method: Using `with_usage=True`

The most effective way to get usage statistics is by setting `with_usage=True` when calling methods like `create_with_completion`. This returns a tuple containing the response, completion, and usage information.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from instructor.usage import UnifiedUsage

client = instructor.from_openai(OpenAI())

class UserExtract(BaseModel):
    name: str
    age: int

user, completion, usage = client.chat.completions.create_with_completion(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
    with_usage=True
)

print(usage)
# UnifiedUsage(total_tokens=91, input_tokens=82, output_tokens=9)
```

The `UnifiedUsage` object provides a consistent interface for usage data across different providers.

## Previous Method: Accessing Raw Response

While still valid, this method may not provide as complete coverage for all providers as the `with_usage=True` method.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class UserExtract(BaseModel):
    name: str
    age: int

user, completion = client.chat.completions.create_with_completion(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(completion.usage)
# CompletionUsage(completion_tokens=9, prompt_tokens=82, total_tokens=91)
```

## Handling Incomplete Output Exceptions

You can catch an `IncompleteOutputException` when the context length is exceeded and react accordingly, such as by trimming your prompt by the number of exceeding tokens.

```python
from instructor.exceptions import IncompleteOutputException
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())

class UserExtract(BaseModel):
    name: str
    age: int

try:
    user, completion, usage = client.chat.completions.create_with_completion(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
        with_usage=True
    )
except IncompleteOutputException as e:
    token_count = e.last_completion.usage.total_tokens  # type: ignore
    # your logic here
```

By using the `with_usage=True` parameter, you can get more consistent and comprehensive usage data across different providers, while still having access to the raw response data if needed.

## Capturing Usage During Exceptions

Instructor allows you to capture usage statistics even when an exception occurs during retries. This is particularly useful for monitoring costs and token usage across all attempts, including failed ones.

Here's an example of how to handle this scenario:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from instructor.exceptions import InstructorRetryException
from instructor.usage import UnifiedUsage

client = instructor.from_openai(OpenAI())

class UserExtract(BaseModel):
    name: str
    age: int

def extract_user_info(prompt: str):
    try:
        user, completion, usage = client.chat.completions.create_with_completion(
            model="gpt-3.5-turbo",
            response_model=UserExtract,
            messages=[{"role": "user", "content": prompt}],
            with_usage=True
        )
        print(f"Successful extraction. Usage: {usage}")
        return user, usage
    except InstructorRetryException as e:
        print(f"Input tokens: {e.total_usage.input_tokens}, Output tokens: {e.total_usage.output_tokens}"
        # Here you could log cost based on the usage
        return None, e.total_usage

# Example usage
prompt = "Extract: Jason is 25 years old"
result, usage = extract_user_info(prompt)

if result:
    print(f"Extracted: {result}")
else:
    print("Extraction failed")

print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Total tokens used: {usage.total_tokens}")
```

In this example:

1. We use `create_with_completion` with `with_usage=True` to get usage statistics.
2. If the extraction succeeds, we return the result and the usage.
3. If an `InstructorRetryException` is caught, we capture the `total_usage` from the exception, which includes the cumulative usage across all retry attempts.
4. Whether the extraction succeeds or fails, we can access and report on the token usage.

This approach allows you to monitor and manage your API usage effectively, even when dealing with retries and potential failures.

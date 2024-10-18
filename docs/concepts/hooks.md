---
title: Understanding Hooks in the Instructor Library
description: Learn how to use hooks for event handling in the Instructor library to enhance logging, error handling, and custom behaviors.
---

# Hooks

Hooks provide a powerful mechanism for intercepting and handling events during the completion and parsing process in the Instructor library. They allow you to add custom behavior, logging, or error handling at various stages of the API interaction.

## Overview

The Hooks system in Instructor is based on the `Hooks` class, which manages event registration and emission. It supports several predefined events that correspond to different stages of the completion and parsing process.

## Supported Hook Events

### `completion:kwargs`

This hook is emitted when completion arguments are provided. It receives all arguments passed to the completion function. These will contain the `model`, `messages`, `tools`, AFTER any `response_model` or `validation_context` parameters have been converted to their respective values.

```python
def handler(*args, **kwargs) -> None: ...
```

### `completion:response`

This hook is emitted when a completion response is received. It receives the raw response object from the completion API.

```python
def handler(response) -> None: ...
```

### `completion:error`

This hook is emitted when an error occurs during completion before any retries are attempted and the response is parsed as a pydantic model.

```python
def handler(error) -> None: ...
```

### `parse:error`

This hook is emitted when an error occurs during parsing of the response as a pydantic model. This can happen if the response is not valid or if the pydantic model is not compatible with the response.

```python
def handler(error) -> None: ...
``` 

### `completion:last_attempt`

This hook is emitted when the last retry attempt is made.

```python
def handler(error) -> None: ...
```

## Implementation Details

The Hooks system is implemented in the `instructor/hooks.py` file. The `Hooks` class handles the registration and emission of hook events. You can refer to this file to see how hooks work under the hood.
The retry logic that uses Hooks is implemented in the `instructor/retry.py` file. This shows how Hooks are used when trying again after errors during completions.

### Registering Hooks

You can register hooks using the `on` method of the Instructor client or a `Hooks` instance. Here's an example:

```python
import instructor
import openai
import pprint

client = instructor.from_openai(openai.OpenAI())


def log_completion_kwargs(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


client.on("completion:kwargs", log_completion_kwargs)

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}],
    response_model=str,
)
print(resp)
#> Hello, world!
```

### Emitting Events

Events are automatically emitted by the Instructor library at appropriate times. You don't need to manually emit events in most cases.

### Removing Hooks

You can remove a specific hook using the `off` method:

```python
# <%hide%>
import instructor
import openai
import pprint

client = instructor.from_openai(openai.OpenAI())
resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}],
    response_model=str,
)


def log_completion_kwargs(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


client.on("completion:kwargs", log_completion_kwargs)
# <%hide%>
client.off("completion:kwargs", log_completion_kwargs)
```

### Clearing Hooks

To remove all hooks for a specific event or all events:

```python
# <%hide%>
import instructor
import openai

client = instructor.from_openai(openai.OpenAI())
resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}],
    response_model=str,
)
# <%hide%>
# Clear hooks for a specific event
client.clear("completion:kwargs")

# Clear all hooks
client.clear()
```

## Example: Logging and Debugging

Here's a comprehensive example demonstrating how to use hooks for logging and debugging:

```python
import instructor
import openai
import pydantic


def log_completion_kwargs(kwargs) -> None:
    print("## Completion kwargs:")
    print(kwargs)
    """
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract the user name and age from the following text: 'John is 20 years old'",
            }
        ],
        "model": "gpt-4o-mini",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "User",
                    "description": "Correctly extracted `User` with all the required parameters with correct types",
                    "parameters": {
                        "properties": {
                            "name": {"title": "Name", "type": "string"},
                            "age": {"title": "Age", "type": "integer"},
                        },
                        "required": ["age", "name"],
                        "type": "object",
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "User"}},
    }
    """


def log_completion_response(response) -> None:
    print("## Completion response:")
    print(response.model_dump())
    """
    {
        "id": "chatcmpl-AJHKkGTSwkxdmxBuaz69q4yCeqIZK",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": [
                        {
                            "id": "call_glxG7L23PiVLHWBT2nxvh4Vs",
                            "function": {
                                "arguments": '{"name":"John","age":20}',
                                "name": "User",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1729158226,
        "model": "gpt-4o-mini-2024-07-18",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": "fp_e2bde53e6e",
        "usage": {
            "completion_tokens": 9,
            "prompt_tokens": 87,
            "total_tokens": 96,
            "completion_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0},
            "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
        },
    }   
    """


def log_completion_error(error) -> None:
    print("## Completion error:")
    print({"error": error})


def log_parse_error(error) -> None:
    print("## Parse error:")
    #> ## Parse error:
    print(error)
    """
    1 validation error for User
    age
    Value error, Age cannot be negative [type=value_error, input_value=-10, input_type=int]
        For further information visit https://errors.pydantic.dev/2.8/v/value_error
    """


# Create an Instructor client
client = instructor.from_openai(openai.OpenAI())

client.on("completion:kwargs", log_completion_kwargs)
client.on("completion:response", log_completion_response)

client.on("completion:error", log_completion_error)
client.on("parse:error", log_parse_error)


# Define a model with a validator
class User(pydantic.BaseModel):
    name: str
    age: int

    @pydantic.field_validator("age")
    def check_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age cannot be negative")
        return v


try:
    # Use the client to create a completion
    user = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Extract the user name and age from the following text: 'John is -1 years old'",
            }
        ],
        response_model=User,
        max_retries=1,
    )
except Exception as e:
    print(f"Error: {e}")


user = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Extract the user name and age from the following text: 'John is 10 years old'",
        }
    ],
    response_model=User,
    max_retries=1,
)
print(user)
#> name='John' age=10
```

This example demonstrates:

1. Defining hook handlers for different events.
2. Registering the hooks with the Instructor client.
3. Using a Pydantic model with a validator.
4. Making a completion request that will trigger various hooks.

The hooks will log information at different stages of the process, helping with debugging and understanding the flow of data.

## Best Practices

1. **Error Handling**: Always include error handling in your hook handlers to prevent exceptions from breaking the main execution flow. We will automatically warn if an exception is raised in a hook handler.

2. **Performance**: Keep hook handlers lightweight to avoid impacting the performance of the main application.

3. **Modularity**: Use hooks to separate concerns. For example, use hooks for logging, monitoring, or custom business logic without cluttering the main code.

4. **Consistency**: Use the same naming conventions and patterns across all your hooks for better maintainability.

5. **Documentation**: Document the purpose and expected input/output of each hook handler for easier collaboration and maintenance.

By leveraging hooks effectively, you can create more flexible, debuggable, and maintainable applications when working with the Instructor library and language models.

# Welcome to Instructor - Your Gateway to Structured Outputs with OpenAI

_Pythonic Structured Outputs powered by LLM function calling and tool calling APIs. Designed for simplicity, transparency, and control._

---

[Star us on Github!](https://www.github.com/jxnl/instructor)

[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/CV8sPM5k5Y)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://jxnl.github.io/instructor)
[![GitHub issues](https://img.shields.io/github/issues/jxnl/instructor.svg)](https://github.com/jxnl/instructor/issues)

Dive into the world of Python-based structured extraction, empowered by OpenAI's cutting-edge function calling API. Instructor stands out for its simplicity, transparency, and user-centric design. Whether you're a seasoned developer or just starting out, you'll find Instructor's approach intuitive and its results insightful.

## Ports to other languages

Check out ports to other languages below:

- [Typescript / Javascript](https://www.github.com/jxnl/instructor-js)
- [Elixir](https://github.com/thmsmlr/instructor_ex/)

If you want to port Instructor to another language, please reach out to us on [Twitter](https://twitter.com/jxnlco) we'd love to help you get started!

## Get Started in Moments

Installing Instructor is a breeze. Simply run `pip install instructor` in your terminal and you're on your way to a smoother data handling experience!

## How Instructor Enhances Your Workflow

Our `instructor.patch` for the `OpenAI` class introduces three key enhancements:

- **Response Mode:** Specify a Pydantic model to streamline data extraction.
- **Max Retries:** Set your desired number of retry attempts for requests.
- **Validation Context:** Provide a context object for enhanced validator access. A Glimpse into Instructor's Capabilities.

### Using Validators

To learn more about validators, checkout our blog post [Good LLM validation is just good validation](https://jxnl.github.io/instructor/blog/2023/10/23/good-llm-validation-is-just-good-validation/)

## Usage

With Instructor, your code becomes more efficient and readable. Here’s a quick peek:

```py hl_lines="5 13"
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Enables `response_model`
client = instructor.patch(OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)

assert isinstance(user, UserDetail)
assert user.name == "Jason"
assert user.age == 25
```

### Using async clients

For async clients you must use `apatch` vs. `patch`, as shown:

```py
import instructor
import asyncio
import openai
from pydantic import BaseModel

aclient = instructor.apatch(openai.AsyncOpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


task = aclient.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)


response = asyncio.run(task)
print(response.model_dump_json(indent=2))
"""
{
  "name": "Jason",
  "age": 25
}
"""
```

### Step 1: Patch the client

First, import the required libraries and apply the `patch` function to the OpenAI module. This exposes new functionality with the `response_model` parameter.

```python
import instructor
from openai import OpenAI

# This enables response_model keyword
# from client.chat.completions.create
client = instructor.patch(OpenAI())
```

### Step 2: Define the Pydantic Model

Create a Pydantic model to define the structure of the data you want to extract. This model will map directly to the information in the prompt.

```python
from pydantic import BaseModel


class UserDetail(BaseModel):
    name: str
    age: int
```

### Step 3: Extract

Use the `client.chat.completions.create` method to send a prompt and extract the data into the Pydantic object. The `response_model` parameter specifies the Pydantic model to use for extraction. It is helpful to annotate the variable with the type of the response model which will help your IDE provide autocomplete and spell check.

```python
import instructor
import openai
from pydantic import BaseModel

client = instructor.patch(openai.OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)

assert isinstance(user, UserDetail)
assert user.name == "Jason"
assert user.age == 25
print(user.model_dump_json(indent=2))
"""
{
  "name": "Jason",
  "age": 25
}
"""
```

## Pydantic Validation

Validation can also be plugged into the same Pydantic model.

In this example, if the answer attribute contains content that violates the rule "Do not say objectionable things", Pydantic will raise a validation error.

```python hl_lines="9 15"
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator


class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str, BeforeValidator(llm_validator("don't say objectionable things"))
    ]


try:
    qa = QuestionAnswer(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and steal",
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for QuestionAnswer
    answer
      Assertion failed, The statement promotes objectionable behavior. [type=assertion_error, input_value='The meaning of life is to be evil and steal', input_type=str]
        For further information visit https://errors.pydantic.dev/2.6/v/assertion_error
    """
```

It is important to note here that the **error message is generated by the LLM**, not the code. Thus, it is helpful for re-asking the model.

```plaintext
1 validation error for QuestionAnswer
answer
   Assertion failed, The statement is objectionable. (type=assertion_error)
```

## Re-ask on validation error

Here, the `UserDetails` model is passed as the `response_model`, and `max_retries` is set to 2.

```python
import instructor

from openai import OpenAI
from pydantic import BaseModel, field_validator

# Apply the patch to the OpenAI client
client = instructor.patch(OpenAI())


class UserDetails(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError("Name must be in uppercase.")
        return v


model = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetails,
    max_retries=2,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(model.model_dump_json(indent=2))
"""
{
  "name": "JASON",
  "age": 25
}
"""
```

## [Evals](https://github.com/jxnl/instructor/tree/main/tests/openai/evals)

We invite you to contribute to evals in `pytest` as a way to monitor the quality of the OpenAI models and the `instructor` library. To get started check out the [jxnl/instructor/tests/evals](https://github.com/jxnl/instructor/tree/main/tests/openai/evals) and contribute your own evals in the form of pytest tests. These evals will be run once a week and the results will be posted.

## Contributing

If you want to help, checkout some of the issues marked as `good-first-issue` or `help-wanted` found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cookbook.

## CLI

We also provide some added CLI functionality for easy convinience:

- `instructor jobs` : This helps with the creation of fine-tuning jobs with OpenAI. Simple use `instructor jobs create-from-file --help` to get started creating your first fine-tuned GPT3.5 model

- `instructor files` : Manage your uploaded files with ease. You'll be able to create, delete and upload files all from the command line

- `instructor usage` : Instead of heading to the OpenAI site each time, you can monitor your usage from the cli and filter by date and time period. Note that usage often takes ~5-10 minutes to update from OpenAI's side

## License

This project is licensed under the terms of the MIT License.

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/jxnl/instructor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxnl/instructor" />
</a>

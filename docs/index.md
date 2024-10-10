# Instructor, The Most Popular Library for Simple Structured Outputs

_Structured outputs powered by llms. Designed for simplicity, transparency, and control._

---

[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/bD9YE9JArw)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)
[![GPT](https://img.shields.io/badge/docs-InstructorGPT-blue)](https://chat.openai.com/g/g-EvZweRWrE-instructor-gpt)

Instructor makes it easy to get structured data like JSON from LLMs like GPT-3.5, GPT-4, GPT-4-Vision, and open-source models including [Mistral/Mixtral](./hub/together.md), [Anyscale](./hub/anyscale.md), [Ollama](./hub/ollama.md), and [llama-cpp-python](./hub/llama-cpp-python.md).

It stands out for its simplicity, transparency, and user-centric design, built on top of Pydantic. Instructor helps you manage [validation context](./concepts/reask_validation.md), retries with [Tenacity](./concepts/retrying.md), and streaming [Lists](./concepts/lists.md) and [Partial](./concepts/partial.md) responses.

[:material-star: Star the Repo](https://github.com/jxnl/instructor){: .md-button .md-button--primary } [:material-book-open-variant: Cookbooks](./examples/index.md){: .md-button } [:material-lightbulb: Prompting Guide](./prompting/index.md){: .md-button }

## Newsletter

If you want to be notified of tips, new blog posts, and research, subscribe to our newsletter. Here's what you can expect:

- Updates on Instructor features and releases
- Blog posts on AI and structured outputs
- Tips and tricks from our community
- Research in the field of LLMs and structured outputs
- Information on AI development skills with Instructor

Subscribe to our newsletter for updates on AI development. We provide content to keep you informed and help you use Instructor in projects.

<iframe src="https://embeds.beehiiv.com/2faf420d-8480-4b6e-8d6f-9c5a105f917a?slim=true" data-test-id="beehiiv-embed" height="52" width="80%" frameborder="0" scrolling="no" style="margin: 0; border-radius: 0px !important; background-color: transparent;"></iframe>

## Why use Instructor?

<div class="grid cards" markdown>

- :material-code-tags: **Simple API with Full Prompt Control**

    Instructor provides a straightforward API that gives you complete ownership and control over your prompts. This allows for fine-tuned customization and optimization of your LLM interactions.

    [:octicons-arrow-right-16: Explore Concepts](./concepts/models.md)

- :material-translate: **Multi-Language Support**

    Simplify structured data extraction from LLMs with type hints and validation.

    [:simple-python: Python](https://python.useinstructor.com) · [:simple-typescript: TypeScript](https://js.useinstructor.com) · [:simple-ruby: Ruby](https://ruby.useinstructor.com) · [:simple-go: Go](https://go.useinstructor.com) · [:simple-elixir: Elixir](https://hex.pm/packages/instructor) · [:simple-rust: Rust](https://rust.useinstructor.com)

- :material-refresh: **Reasking and Validation**

    Automatically reask the model when validation fails, ensuring high-quality outputs. Leverage Pydantic's validation for robust error handling.

    [:octicons-arrow-right-16: Learn about Reasking](./concepts/reask_validation.md)

- :material-repeat-variant: **Streaming Support**

    Stream partial results and iterables with ease, allowing for real-time processing and improved responsiveness in your applications.

    [:octicons-arrow-right-16: Learn about Streaming](./concepts/partial.md)

- :material-code-braces: **Powered by Type Hints**

    Leverage Pydantic for schema validation, prompting control, less code, and IDE integration.

    [:octicons-arrow-right-16: Learn more](https://docs.pydantic.dev/)

- :material-lightning-bolt: **Simplified LLM Interactions**

    Support for [OpenAI](./hub/openai.md), [Anthropic](./hub/anthropic.md), [Google](./hub/google.md), [Vertex AI](./hub/vertexai.md), [Mistral/Mixtral](./hub/together.md), [Anyscale](./hub/anyscale.md), [Ollama](./hub/ollama.md), [llama-cpp-python](./hub/llama-cpp-python.md), [Cohere](./hub/cohere.md), [LiteLLM](./hub/litellm.md).

    [:octicons-arrow-right-16: See Hub](./hub/index.md)

</div>

## Getting Started

```
pip install -U instructor
```

If you ever get stuck, you can always run `instructor docs` to open the documentation in your browser. It even supports searching for specific topics.

```
instructor docs [QUERY]
```

You can also check out our [cookbooks](./examples/index.md) and [concepts](./concepts/models.md) to learn more about how to use Instructor.

??? info "Make sure you've installed the dependencies for your specific client"

    To keep the bundle size small, `instructor` only ships with the OpenAI client. Before using the other clients and their respective `from_xx` method, make sure you've installed the dependencies following the instructions below.

    1. Anthropic : `pip install "instructor[anthropic]"`
    2. Google Generative AI: `pip install "instructor[google-generativeai]"`
    3. Vertex AI: `pip install "instructor[vertexai]"`
    4. Cohere: `pip install "instructor[cohere]"`
    5. Litellm: `pip install "instructor[litellm]"`
    6. Mistral: `pip install "instructor[mistralai]"`

Now, let's see Instructor in action with a simple example:

### Using OpenAI

??? info "Want to use OpenAI's Structured Output Response?"

    We've added support for OpenAI's structured output response. With this, you'll get all the benefits of instructor you like with the constrained sampling from OpenAI.

    ```python
    from openai import OpenAI
    from instructor import from_openai, Mode
    from pydantic import BaseModel

    client = from_openai(OpenAI(), mode=Mode.TOOLS_STRICT)


    class User(BaseModel):
        name: str
        age: int


    resp = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        model="gpt-4o",
    )
    ```

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(user_info.name)
#> John Doe
print(user_info.age)
#> 30
```

### Using Anthropic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_anthropic(Anthropic())

# note that client.chat.completions.create will also work
resp = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

### Using Gemini

The Vertex AI and Gemini Clients have different APIs. When using instructor with these clients, make sure to read the documentation for the specific client you're using to make sure you're using the correct methods.

**Note**: Gemini Tool Calling is still in preview, and there are some limitations. You can learn more about them in the [Vertex AI examples notebook](../hub/vertexai.md). As of now, you cannot use tool calling with Gemini when you have multi-modal inputs (Eg. Images, Audio, Video), you must use the `JSON` mode equivalent for that client.

#### Google AI

```python
import instructor
import google.generativeai as genai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)

# note that client.chat.completions.create will also work
resp = client.messages.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

??? info "Using Gemini's multi-modal capabilities with `google-generativeai`"

    The `google.generativeai` library has a different API than the `vertexai` library. But, using `instructor`, working with multi-modal data is easy.

    Here's a quick example of how to use an Audio file with `google-generativeai`. We've used this [recording](https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3) that's taken from the [Google Generative AI cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb)

    For a more in-depth example, you can check out our guide to working with Gemini using the `google-generativeai` package [here](./examples/multi_modal_gemini.md).


    ```python
    import instructor
    import google.generativeai as genai
    from pydantic import BaseModel


    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest",
        ),
        mode=instructor.Mode.GEMINI_JSON,  # (1)!
    )

    mp3_file = genai.upload_file("./sample.mp3")  # (2)!


    class Description(BaseModel):
        description: str


    resp = client.create(
        response_model=Description,
        messages=[
            {
                "role": "user",
                "content": "Summarize what's happening in this audio file and who the main speaker is",
            },
            {
                "role": "user",
                "content": mp3_file,  # (3)!
            },
        ],
    )

    print(resp)
    #> description="The main speaker is President John F. Kennedy, and he's giving a
    #> State of the Union address to a joint session of Congress. He begins by
    #> acknowledging his fondness for the House of Representatives and his long
    #> history with it. He then goes on to discuss the state of the economy,
    #> highlighting the difficulties faced by Americans, such as unemployment and
    #> low farm incomes. He also touches on the Cold War and the international
    #> balance of payments. He speaks of the need to strengthen the US military,
    #> and he also discusses the importance of international cooperation and the
    #> need to address global issues like hunger and illiteracy. He ends by urging
    #> his audience to work together to face the challenges that lie ahead."
    ```

    1. Make sure to set the mode to `GEMINI_JSON`, this is important because Tool Calling doesn't work with multi-modal inputs.
    2. Use `genai.upload_file` to upload your file. If you've already uploaded the file, you can get it by using `genai.get_file`
    3. Pass in the file object as any normal user message

#### Vertex AI

```python
import instructor
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
from pydantic import BaseModel

vertexai.init()


class User(BaseModel):
    name: str
    age: int


client = instructor.from_vertexai(
    client=GenerativeModel("gemini-1.5-pro-preview-0409"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
)

# note that client.chat.completions.create will also work
resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

??? info "Using Gemini's multi-modal capabilities with VertexAI"

    We've most recently added support for multi-part file formats using google's `gm.Part` objects. This allows you to pass in additional information to the LLM about the data you'd like to see.

    Here are two examples of how to use multi-part formats with Instructor.

    We can combine multiple `gm.Part` objects into a single list and combine them into a single message to be sent to the LLM. Under the hood, we'll convert them into the appropriate format for Gemini.

    ```python
    import instructor
    import vertexai.generative_models as gm  # type: ignore
    from pydantic import BaseModel, Field

    client = instructor.from_vertexai(gm.GenerativeModel("gemini-1.5-pro-001"))
    content = [
        "Order Details:",
        gm.Part.from_text("Customer: Alice"),
        gm.Part.from_text("Items:"),
        "Name: Laptop, Price: 999.99",
        "Name: Mouse, Price: 29.99",
    ]


    class Item(BaseModel):
        name: str
        price: float


    class Order(BaseModel):
        items: list[Item] = Field(..., default_factory=list)
        customer: str


    resp = client.create(
        response_model=Order,
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    print(resp)
    #> items=[Item(name='Laptop', price=999.99), Item(name='Mouse', price=29.99)] customer='Alice'
    ```

    This is also the same for multi-modal responses when we want to work with images. In this example, we'll ask the LLM to describe an image and pass in the image as a `gm.Part` object.

    ```python
    import instructor
    import vertexai.generative_models as gm  # type: ignore
    from pydantic import BaseModel
    import requests

    client = instructor.from_vertexai(
        gm.GenerativeModel("gemini-1.5-pro-001"), mode=instructor.Mode.VERTEXAI_JSON
    )
    content = [
        gm.Part.from_text("Count the number of objects in the image."),
        gm.Part.from_data(
            bytes(
                requests.get(
                    "https://img.taste.com.au/Oq97xT-Q/taste/2016/11/blueberry-scones-75492-1.jpeg"
                ).content
            ),
            "image/jpeg",
        ),
    ]


    class Description(BaseModel):
        description: str


    resp = client.create(
        response_model=Description,
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    print(resp)
    #> description='Seven blueberry scones sit inside a metal pie plate.'
    ```

### Using Litellm

```python
import instructor
from litellm import completion
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_litellm(completion)

resp = client.chat.completions.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

### Using Cohere

We also support users who want to use the Cohere models using the `from_cohere` method.

??? info "Want to get the original Cohere response?"

    If you want to get the original response object from the LLM instead of a structured output, you can pass `response_model=None` to the `create` method. This will return the raw response from the underlying API.

    ```python
    # This will return the original Cohere response object
    raw_response = client.chat.completions.create(
        response_model=None,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
    )
    ```

    This can be useful when you need access to additional metadata or want to handle the raw response yourself.

```python
import instructor
from pydantic import BaseModel
from cohere import Client


class User(BaseModel):
    name: str
    age: int


client = instructor.from_cohere(Client())

resp = client.chat.completions.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
)

assert resp.name == "Jason"
assert resp.age == 25
```

### Using Cerebras

For those who want to use the Cerebras models, you can use the `from_cerebras` method to patch the client. You can see their list of models [here](https://inference-docs.cerebras.ai/api-reference/models).

```python
from cerebras.cloud.sdk import Cerebras
import instructor
from pydantic import BaseModel
import os

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)
client = instructor.from_cerebras(client)


class User(BaseModel):
    name: str
    age: int


resp = client.chat.completions.create(
    model="llama3.1-70b",
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
)

print(resp)
#> name='Jason' age=25
```

### Using Fireworks

For those who want to use the Fireworks models, you can use the `from_fireworks` method to patch the client. You can see their list of models [here](https://fireworks.ai/models).

```python
from fireworks.client import Fireworks
import instructor
from pydantic import BaseModel
import os

client = Fireworks(
    api_key=os.environ.get("FIREWORKS_API_KEY"),
)
client = instructor.from_fireworks(client)


class User(BaseModel):
    name: str
    age: int


resp = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p2-1b-instruct",
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
)

print(resp)
#> name='Jason' age=25
```

## Correct Typing

This was the dream of instructor but due to the patching of openai, it wasnt possible for me to get typing to work well. Now, with the new client, we can get typing to work well! We've also added a few `create_*` methods to make it easier to create iterables and partials, and to access the original completion.

### Calling `create`

```python
import openai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_openai(openai.OpenAI())

user = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)
```

Now if you use a IDE, you can see the type is correctly infered.

![type](./blog/posts/img/type.png)

### Handling async: `await create`

This will also work correctly with asynchronous clients.

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.AsyncOpenAI())


class User(BaseModel):
    name: str
    age: int


async def extract():
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=User,
    )
```

Notice that simply because we return the `create` method, the `extract()` function will return the correct user type.

![async](./blog/posts/img/async_type.png)

### Returning the original completion: `create_with_completion`

You can also return the original completion object

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


user, completion = client.chat.completions.create_with_completion(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)
```

![with_completion](./blog/posts/img/with_completion.png)

### Streaming Partial Objects: `create_partial`

In order to handle streams, we still support `Iterable[T]` and `Partial[T]` but to simply the type inference, we've added `create_iterable` and `create_partial` methods as well!

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


user_stream = client.chat.completions.create_partial(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)

for user in user_stream:
    print(user)
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name='John Doe' age=25
    # name=None age=None
    # name='' age=None
    # name='John' age=None
    # name='John Doe' age=None
    # name='John Doe' age=30
```

Notice now that the type infered is `Generator[User, None]`

![generator](./blog/posts/img/generator.png)

### Streaming Iterables: `create_iterable`

We get an iterable of objects when we want to extract multiple objects.

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


users = client.chat.completions.create_iterable(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create 2 users"},
    ],
    response_model=User,
)

for user in users:
    print(user)
    #> name='John Doe' age=30
    #> name='Jane Doe' age=28
    # User(name='John Doe', age=30)
    # User(name='Jane Smith', age=25)
```

![iterable](./blog/posts/img/iterable.png)

## Templating

Instructor also ships with [Jinja](https://palletsprojects.com/p/jinja/) templating support. Check out our docs on [templating](./concepts/templating.md) to learn about how to use it to its full potential.

## Validation

You can also use Pydantic to validate your outputs and get the llm to retry on failure. Check out our docs on [retrying](./concepts/retrying.md) and [validation context](./concepts/reask_validation.md).

## More Examples

If you'd like to see more check out our [cookbook](examples/index.md).

[Installing Instructor](installation.md) is a breeze. Just run `pip install instructor`.

## Contributing

If you want to help out, checkout some of the issues marked as `good-first-issue` or `help-wanted`. Found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cook book.

## License

This project is licensed under the terms of the MIT License.

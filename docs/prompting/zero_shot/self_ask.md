---
description: "Self-Ask is a prompting technique that enhances language model performance by encouraging the model to generate and answer follow-up questions before tackling the main query, leading to more accurate and comprehensive responses."
---

By encouraging our model to generate and answer clarifying questions before tackling the main query, we can obtain more accurate and comprehensive responses. This is known as Self-Ask <sup><a href="https://arxiv.org/pdf/2210.03350">1</a></sup>.

We can implement this using `instructor` easily as seen below

```python hl_lines="44-46"
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class FollowupQuestion(BaseModel):
    question: str = Field(
        ...,
        description="""Question to be answered""",
    )
    answer: str = Field(
        ...,
        description="Answer to the Follow Up Question",
    )


class SelfAskResponse(BaseModel):
    follow_up_questions: list[FollowupQuestion] = Field(
        ...,
        description="""A list of question and
            answer pairs that are required to be
            answered in order to answer the original
            question.""",
        default_factory=list,
    )
    answer: str = Field(
        ...,
        description="Answer to the user's question",
    )


def generate_response(query):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=SelfAskResponse,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant
                which is able to accurately answer user
                queries. Generate a list of follow up
                questions and responses before answering
                the user's question below.""",
            },
            {
                "role": "user",
                "content": f"{query}",
            },
        ],
    )


if __name__ == "__main__":
    query = """Who was president of the U.S.
    when superconductivity was discovered?"""

    response = generate_response(query)
    print(response.follow_up_questions)
    """
    [
        FollowupQuestion(
            question='When was superconductivity discovered?',
            answer='Superconductivity was discovered in April 1911.',
        ),
        FollowupQuestion(
            question='Who was the president of the U.S. in April 1911?',
            answer='The President of the U.S. in April 1911 was William Howard Taft.',
        ),
    ]
    """
    print(response.answer)
    """
    The President of the U.S. when superconductivity was discovered in April 1911
    was William Howard Taft.
    """
```

### References

<sup id="ref-1">1</sup>: [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/pdf/2210.03350)

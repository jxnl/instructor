---
description: "Tab-CoT encourages LLMs to output reasoning as a markdown table, improving the structure and reasoning of its output"
---

By getting language models to output their reasoning as a structured markdown table, we can improve their reasoning capabilities and the quality of their outputs. This is known as Tabular Chain Of Thought (Tab-CoT) <sup><a href="https://arxiv.org/pdf/2305.17812">1</a></sup>.

We can implement this using `instructor` as a response object as seen below to ensure we get exactly the data that we want. Each row in our table is represented here as a `ReasoningStep` object.

```python linenums="1"
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class ReasoningStep(BaseModel):
    step: int = Field(..., description="The step number")
    subquestion: str = Field(..., description="Subquestion to solve")
    procedure: str = Field(
        ...,
        description="""Any intermediate computation
        that was done in the reasoning process. Leave
        empty if no computation is needed""",
    )
    result: str = Field(
        ...,
        description="Final Answer",
    )


class Response(BaseModel):
    reasoning: list[ReasoningStep] = Field(
        ...,
        description="reasoning steps to derive answer",
    )
    answer: str = Field(
        ...,
        description="Final answer",
    )


def generate_structured_reasoning_response(query: str, context: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": f"""
                {context}
                {query}
                """,
            },
        ],
    )
    return response


if __name__ == "__main__":
    query = "How many loaves of bread did they have left?"
    context = """
    The bakers at the Beverly Hills Bakery baked
    200 loaves of bread on Monday morning. They
    sold 93 loaves in the morning and 39 loaves
    in the afternoon. A grocery store returned 6
    unsold loaves.
    """

    response = generate_structured_reasoning_response(query, context)
    print(f"\nAnswer: {response.answer}")
    """
    Answer: 74
    """
```

This generates the following reasoning step and the correct response of 74.

| Step | Subquestion                                                                                   | Procedure | Result |
| ---- | --------------------------------------------------------------------------------------------- | --------- | ------ |
| 1    | How many loaves of bread were sold in total on Monday?                                        | 93 + 39   | 132    |
| 2    | How many loaves of bread were left after accounting for loaves sold?                          | 200 - 132 | 68     |
| 3    | How many loaves of bread were left after accounting for loaves returned by the grocery store? | 68 + 6    | 74     |

Answer: 74

### References

<sup id="ref-1">1</sup>: [Tab-CoT: Zero-shot Tabular Chain of Thought](https://arxiv.org/pdf/2305.17812)

---
description: "Rephrase and Respond aims to reduce prompt ambiguity and align the question more closely with the LLM's existing frame"
---

We can improve performance of our LLM by getting the model to rewrite the prompt<sup><a href="https://arxiv.org/pdf/2311.04205">1</a></sup> such that is is less ambigious.

This could look something like this

!!! example "Rephrase and Respond Example"

    **User**: Take the last letters of the words in 'Edgar Bob' and concatenate them.

    **Rephrased Question**: Could you please form a new string or series of characters by joining together the final letters from each word in the phrase "Edgar Bob"?

    **Assistant**: The last letters in the words "Edgar" and "Bob" are "r" and "b", hence when concatenated, it forms "rb".

We can implement this using `instructor` as seen below.

```python hl_lines="24-26"
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())


class ImprovedQuestion(BaseModel):
    rewritten_question: str = Field(
        ..., description="""An improved, more specific
        version of the original question"""
    )


class FinalResponse(BaseModel):
    response: str = Field(..., description="Answer to user's question")


def rewrite_question(question: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You excel at making questions clearer
                and more specific.""",
            },
            {"role": "user", "content": f"The question is {question}"},
        ],
        response_model=ImprovedQuestion,
    )


def answer_question(question: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        max_tokens=1000,
        response_model=FinalResponse,
    ).response


if __name__ == "__main__":
    rewritten_query = rewrite_question(
        "Take the last letters of the words in 'Elon Musk' and concatenate them"
    )
    print(f"Rewritten query: {rewritten_query.rewritten_question}")
    """
    Rewritten query: What is the result when you take the last letters of each
    word in the name 'Elon Musk' and concatenate them?
    """

    response = answer_question(rewritten_query.rewritten_question)
    print(f"Response: {response}")
    #> Response: nk
```

We can also achieve the same benefits by **using a better model to generate the question** before we prompt a weaker model - this is known as a two-step RaR.

## Useful Tips

Here are some phrases that you can add to your prompt to refine the question before you generate a response

- Reword and elaborate on the inquiry, then provide an answer.
- Reframe the question with additional context and detail, then provide an answer.
- Modify the original question for clarity and detail, then offer an answer.
- Restate and elaborate on the inquiry before proceeding with a response.
- Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.

### References

<sup id="ref-1">1</sup>: [Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves](https://arxiv.org/pdf/2311.04205)

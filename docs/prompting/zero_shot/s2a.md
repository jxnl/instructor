---
description: "System 2 Attention (S2A) is a two-step prompting technique that focuses on improving the LLM's attention to relevant information"
---

Refine your prompt using the System 2 Attention (S2A) technique<sup><a href="https://arxiv.org/abs/2311.11829">1</a></sup>.

1. Ask the LLM to rewrite the prompt by removing any information unrelated to the question.
2. Pass this new, focused prompt back to the LLM to generate the final response.

This method helps in producing more factual and less opinionated outputs<sup><a href="https://arxiv.org/abs/2311.11829">1</a></sup>.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>. We can implement this using `instructor` as seen below.

```python hl_lines="25-28"
import openai
import instructor
from pydantic import BaseModel, Field

client = instructor.from_openai(openai.OpenAI())


class Step1(BaseModel):
    relevant_context: str = Field(..., description="Relevant context")
    user_query: str = Field(..., description="The question from the user")


class Step2(BaseModel):
    answer: int


def rewrite_prompt():
    rewritten_prompt = client.chat.completions.create(
        model="gpt-4o",
        response_model=Step1,
        messages=[
            {
                "role": "user",
                "content": f"""
                    Given the following text by a user, extract the part
                    that is actually relevant to their question. Please
                    include the actual question or query that the user
                    is asking.

                    Text by user:
                    Mary has 3 times as much candy as Megan. Mary then
                    adds 10 more pieces of candy to her collection. Max
                    is 5 years older than Mary. If Megan has 5 pieces of
                    candy, how many does Mary have in total?
                    """,
            }
        ],
    )
    return rewritten_prompt


def generate_final_response(rewritten_prompt):
    final_response = client.chat.completions.create(
        model="gpt-4o",
        response_model=Step2,
        messages=[
            {
                "role": "user",
                "content": f"""{rewritten_prompt.relevant_context}
                    Question: {rewritten_prompt.user_query}""",
            }
        ],
    )
    return final_response


if __name__ == "__main__":
    # Step 1: Rewrite the prompt
    rewritten_prompt = rewrite_prompt()
    print(rewritten_prompt.relevant_context)
    """
    Mary has 3 times as much candy as Megan.
    Mary then adds 10 more pieces of candy to her collection.
    If Megan has 5 pieces of candy, how many does Mary have in total?
    """
    print(rewritten_prompt.user_query)
    #> how many does Mary have in total?

    # Step 2: Generate the final response
    final_response = generate_final_response(rewritten_prompt)
    print(final_response.answer)
    #> 25
```

### References

<sup id="ref-1">1</sup>: [System 2 Attention (is something you might need too)](https://arxiv.org/abs/2311.11829)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)

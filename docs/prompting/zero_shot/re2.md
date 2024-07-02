---
description: "We can see a small improvement of <4% in different models by just appending the phrase - Read The Question Again."
---

# Read the Prompt Again

By appending the phrase "Read the question again", you can improve the reasoning abilities of Large Langauge Models <sup><a href="https://arxiv.org/pdf/2309.06275">1</a></sup>

This could look something like this

!!! example "Re-Reading Template"

    **[ Input Query ]**
    Read the question again: **[ Input Query ]**

    **[ Critical Thinking Prompt  ]**

We can implement this using `instructor` as seen below.

```python hl_lines="20-21"
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class Solution(BaseModel):
    final_answer: int = Field(..., description="Final Answer")


def solve_question(question: str) -> int:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=Solution,
        messages=[
            {
                "role": "system",
                "content": f"""{question}. Read the question
                again. {question}. Adhere to the provided
                format when responding to the problem and
                make sure to think through this step by
                step.""",
            },
        ],
    )
    return response.final_answer


# Example usage
if __name__ == "__main__":
    question = """Roger has 5 tennis balls. He buys 2 more cans of tennis
    balls. Each can has 3 tennis balls. How many tennis balls
    does he have now?"""

    answer = solve_question(question)
    print(answer)
    #> 11
```

### References

<sup id="ref-1">1</sup>: [Re-Reading Improves Reasoning in Large Language Models](https://arxiv.org/pdf/2309.06275)

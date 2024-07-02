---
description: "Tip: Use SimToM to help the LLM answer tricky questions about different people or things in two easy steps"
---

We can handle complex questions that deal with different perspectives with a two-step process. This involves

1. Encouraging the LLM to identify and isolate the relevant information
2. Instructing the LLM to take a specific perspective and answer the question doing so.

This helps our model focus on the most pertinent details and ultimately produce more accurate and relevant responses.

!!! example "Sample Template"

    **Step 1**: Given the following context, list the facts that {entity} would know. Context: {context}

    **Step 2**: You are {entity}. Answer the following question based only on these facts you know {facts}. Question: {question}

This approach can help eliminate the influence of irrelevant information in the prompt.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>. We can implement this using `instructor` as seen below.

```python hl_lines="24-25"
import openai
import instructor
from pydantic import BaseModel
from typing import Iterable

client = instructor.from_openai(openai.OpenAI())


class KnownFact(BaseModel):
    fact: str


class Response(BaseModel):
    location: str


def generate_known_facts(entity: str) -> Iterable[KnownFact]:
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Iterable[KnownFact],
        messages=[
            {
                "role": "user",
                "content": f"""Given the following context, list
                the facts that {entity} would know:

                Context:
                Alice puts the book on the table.
                Alice leaves the room.
                Bob moves the book to the shelf.
                Where does {entity} think the book is?

                List only the facts relevant to {entity}.
                """,
            }
        ],
    )


def answer_question_based_on_facts(
    entity: str, known_facts: Iterable[KnownFact]
) -> Response:
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": f"""You are {entity}. Answer the following question
                based only on these facts you know:
                {" ".join([str(fact) for fact in known_facts])}""",
            },
            {
                "role": "user",
                "content": f"Question: Where does {entity} think the book is?",
            },
        ],
    )


if __name__ == "__main__":
    known_facts = generate_known_facts("Alice")
    response = answer_question_based_on_facts("Alice", known_facts)

    for fact in known_facts:
        print(fact)
        #> fact='Alice puts the book on the table.'
        #> fact='Alice leaves the room.'
    print(response.location)
    #> on the table
```

### References

<sup id="ref-1">1</sup>: [Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities](https://arxiv.org/abs/2311.10227)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)

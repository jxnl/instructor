---
description: "Role prompting is a technique where we assign a specific role to a LLM. We can do so by using the format"
---

By assigning a specific role to the model, we can improve the performance of the model. This can be done by using the format below.

!!! example "Role Prompting Template"

    You are a **[ role ]**. You **[ description of task ]**. **[ Reiterate instructions ]**.

We can implement this using `instructor` as seen below.

```python hl_lines="23-26"
import openai
import instructor
from typing import Literal
from pydantic import BaseModel, Field

client = instructor.from_openai(openai.OpenAI())


class Label(BaseModel):
    label: Literal["TECHNICAL", "PRODUCT", "BILLING"] = Field(
        ...,
        description="A label that best desscribes the support ticket",
    )


def classify(support_ticket_title: str):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Label,
        messages=[
            {
                "role": "system",
                "content": f"""You are a support agent at a tech company.
                You will be assigned a support ticket to classify.
                Make sure to only select the label that applies to
                the support ticket.""",
            },
            {
                "role": "user",
                "content": f"Classify this ticket: {support_ticket_title}",
            },
        ],
    )


if __name__ == "__main__":
    label_prediction = classify(
        "My account is locked and I can't access my billing info"
    )
    print(label_prediction.label)
    #> BILLING
```

!!! note "This is an example of Role Based Prompting"

    - **Role**: You are a support agent at a tech company
    - **Task** : You will be assigned a support ticket to classify
    - **Reminder**: Make sure to only select the label that applies to the support ticket

### References

<sup id="ref-1">1</sup>: [RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Lanuage Models](https://arxiv.org/abs/2310.00746)  
<sup id="ref-2">2</sup>: [Is "A Helpful Assistant" the Best Role for Large Language Models? A Systematic Evaluation of Social Roles in System Prompts ](https://arxiv.org/abs/2311.10054)  
<sup id="ref-3">3</sup>: [Douglas C. Schmidt, Jesse Spencer-Smith, Quchen Fu, and Jules White. 2023. Cataloging prompt patterns to enhance the discipline of prompt engineering. Dept. of Computer Science, Vanderbilt University.](https://www.dre.vanderbilt.edu/~schmidt/PDF/ADA_Europe_Position_Paper.pdf)  
<sup id="ref-4">4</sup>: [Unleashing the Emergent Cognitive Synergy in Large Lanuage Models: A Task-Solving Agent through Multi-Persona Self-Collaboration ](https://arxiv.org/abs/2307.05300)  
<sup id="ref-5">5</sup>: [Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://dl.acm.org/doi/10.1145/3411763.3451760)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)

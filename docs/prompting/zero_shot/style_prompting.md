---
description: "Use a specific style and be specific about the tone, pacing and other aspects of the model's response"
---

Be specific about the desired writing style<sup><a href="https://arxiv.org/abs/2302.09185">1</a></sup> that you want from the model using specific instructions to the model about tone, pacing among other factors.

By giving clear directions about these stylistic elements, you can guide the model to produce text that closely matches your intended style and format. We can implement this using `instructor` as seen below.

```python hl_lines="19-23"
import instructor
from pydantic import BaseModel
import openai


class Email(BaseModel):
    message: str


client = instructor.from_openai(openai.OpenAI())


def create_email():
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": """Write an email addressed to Mr. John Smith
                from Jane Doe. The email should be formal and the
                message should be polite and respectful. The topic of
                the email is to invite Mr Smith to a business meeting
                on July 15, 2024 at 10:00 AM in Room 123.""",
            }
        ],
        response_model=Email,
    )
    return email.message

if __name__ == "__main__":
    email_message = create_email()
    print(email_message)
    """
    Subject: Invitation to Business Meeting

    Dear Mr. John Smith,

    I hope this message finds you well. My name is Jane Doe, and I am
    writing to cordially invite you to a business meeting that will be
    held on July 15, 2024, at 10:00 AM in Room 123.

    Your presence at this meeting would be greatly valued as we plan to
    discuss important matters pertaining to our ongoing projects and
    future collaborations. We believe your insights and experience will
    greatly contribute to the success of our discussions.

    Please let me know if this time and date are convenient for you, or
    if any adjustments are necessary. We look forward to your
    affirmative response.

    Thank you for considering this invitation.

    Warm regards,
    Jane Doe
    """
```

## Useful Tips

Here are some useful phrases that you might want to include when using style prompting to get the desired output that you want.

| Category         | Template                                                                                                                                                                                                                                                                                                     | Example Phrases                                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Writing Style    | 1. Write a **style** passage about **topic**<br>2. Write a passage with a **style** writing style about **topic**<br>3. Write a **style** passage about **topic**                                                                                                                                            | **style**: functional, flowery, candid, prosaic, ornate, or poetic<br>**topic**: sunsets, strawberries, or writing a paper                    |
| Tone             | 1. Write a **tone** passage about **subject**<br>2. Write a passage about **subject** with a **tone** tone<br>3. Create a **tone** passage about **subject**                                                                                                                                                 | **tone**: dramatic, humorous, optimistic, sad<br>**subject**: love, life, humanity                                                            |
| Emotional Tone   | 1. Write a passage about **subject** that makes the reader feel **emotion**<br>2. Write a passage about **subject** with a **emotion** mood<br>3. Create a passage about **subject** that makes the reader feel **emotion**                                                                                  | **subject**: love, life, humanity<br>**emotion**: angry, fearful, happy, sad, envious, anxious, proud, regretful, surprised, loved, disgusted |
| Characterization | 1. Write a story about **subject** with indirect characterization<br>2. Write a story about **subject** with direct characterization<br>3. Create a story about **subject** where the characters are described indirectly<br>4. Create a story about **subject** where the characters are described directly | **subject**: lovers, cats, survivors                                                                                                          |
| Pacing           | 1. Write a **pace**-paced story about **subject**<br>2. Write a story about **subject** that is **pace**-paced<br>3. Create a **pace**-paced story about **subject**                                                                                                                                         | **pace**: fast, slow<br>**subject**: lovers, cats, survivors                                                                                  |

## References

<sup id="ref-1">1</sup>: [Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints](https://arxiv.org/abs/2302.09185)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)

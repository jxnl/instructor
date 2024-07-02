---
description: "Using emotional language, we can improve the results of our LLM calls and encourage more open-ended text generation"
---

Use emotional language in prompts to enhance the performance of language models. This includes phrases such as

- This is very important for my career
- Are you sure?
- Are you sure that's your final answer? It might be worth taking another look.

We can implement this using `instructor` as seen below.

```python hl_lines="25"
import openai
import instructor
from pydantic import BaseModel
from typing import Iterable


class Album(BaseModel):
    name: str
    artist: str
    year: int


client = instructor.from_openai(openai.OpenAI())


def get_albums():
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Iterable[Album],
        messages=[
            {
                "role": "user",
                "content": """
                Provide me a list of 3 musical albums from the 2000s.
                This is very important to my career.""",  # (1)!
            }
        ],
    )


if __name__ == "__main__":
    albums = get_albums()
    for album in albums:
        print(album)
        #> name='Kid A' artist='Radiohead' year=2000
        #> name='Stankonia' artist='OutKast' year=2000
        #> name='Is This It' artist='The Strokes' year=2001
```

1.  The phrase `This is very important to my career` is a simple example of a sentence that uses emotion prompting.

### Useful Tips

These are some phrases which you can append today to your prompt to use emotion prompting.

1. Write your answer and give me a confidence score between 0-1 for your answer.
2. This is very important to my career.
3. You'd better be sure.
4. Are you sure?
5. Are you sure that's your final answer? Believe in your abilities and strive for excellence. Your hard work will yield remarkable results.
6. Embrace challenges as opportunities for growth. Each obstacle you overcome brings you closer to success.
7. Stay focused and dedicated to your goals. Your consistent efforts will lead to outstanding achievements.
8. Take pride in your work and give it your best. Your commitment to excellence sets you apart.
9. Remember that progress is made one step at a time. Stay determined and keep moving forward.

We want phrases that either

- **Encourage Self Monitoring** : These are phrases that generally encourage humans to reflect on their responses (e.g., Are you sure?) in anticipation of the judgement of others
- **Set a higher bar** : These are phrases that generally encourage humans to have a higher standard for themselves (e.g., Take pride in your work and give it your best shot).
- **Reframe the task**: These are phrases that typically help people to help see the task in a more positive and objective manner (e.g., Are you sure that's your final answer? Believe in your abilities and strive for excellence. Your hard work will yield remarkable results).

### References

<sup id="ref-1">1</sup>: [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)

---
title: Seamless Multimodal Interactions with Instructor
description: Learn how the Image class in Instructor enables seamless handling of images and text across different AI models.
---

# Multimodal

Instructor supports multimodal interactions by providing helper classes that are automatically converted to the correct format for different providers, allowing you to work with both text and images in your prompts and responses. This functionality is implemented in the `multimodal.py` module and provides a seamless way to handle images alongside text for various AI models.

## `Image`

The core of multimodal support in Instructor is the `Image` class. This class represents an image that can be loaded from a URL or file path. It provides methods to create `Image` instances and convert them to formats compatible with different AI providers.

It's important to note that Anthropic and OpenAI have different formats for handling images in their API requests. The `Image` class in Instructor abstracts away these differences, allowing you to work with a unified interface.

### Usage

You can create an `Image` instance from a URL or file path using the `from_url` or `from_path` methods. The `Image` class will automatically convert the image to a base64-encoded string and include it in the API request.

```python
import instructor
import openai
from pydantic import BaseModel


class ImageAnalyzer(BaseModel):
    description: str


# <%hide%>
import requests

url = "https://static01.nyt.com/images/2017/04/14/dining/14COOKING-RITZ-MUFFINS/14COOKING-RITZ-MUFFINS-jumbo.jpg"
response = requests.get(url)

with open("muffin.jpg", "wb") as file:
    file.write(response.content)
# <%hide%>

image1 = instructor.Image.from_url(url)
image2 = instructor.Image.from_path("muffin.jpg")

client = instructor.from_openai(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ImageAnalyzer,
    messages=[
        {"role": "user", "content": ["What is in this two images?", image1, image2]}
    ],
)

print(response.model_dump_json())
"""
{"description":"A tray of blueberry muffins, some appear whole while one is partially broken showing its soft texture, all have golden-brown tops and are placed on a delicate, patterned surface."}
"""
```

The `Image` class takes care of the necessary conversions and formatting, ensuring that your code remains clean and provider-agnostic. This flexibility is particularly valuable when you're experimenting with different models or when you need to switch providers based on specific project requirements.

By leveraging Instructor's multimodal capabilities, you can focus on building your application logic without worrying about the intricacies of each provider's image handling format. This not only saves development time but also makes your code more maintainable and adaptable to future changes in AI provider APIs.
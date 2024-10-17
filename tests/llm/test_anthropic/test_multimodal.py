import pytest
from instructor.multimodal import Image
import instructor
from pydantic import Field, BaseModel
from itertools import product
from .util import models, modes


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    Image.from_url(
                        "https://pbs.twimg.com/profile_images/1816950591857233920/ZBxrWCbX_400x400.jpg"
                    ),
                ],
            },
        ],
        temperature=1,
        max_tokens=1000,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    "https://pbs.twimg.com/profile_images/1816950591857233920/ZBxrWCbX_400x400.jpg",
                ],
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_image_params(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    {
                        "type": "image",
                        "source": "https://pbs.twimg.com/profile_images/1816950591857233920/ZBxrWCbX_400x400.jpg",
                    },
                ],
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_image_params_cache(
    model, mode, client
):
    client = instructor.from_anthropic(client, mode=mode, enable_prompt_caching=True)
    messages = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=None,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images and stuff",
            },
            {
                "role": "user",
                "content": [
                    "Describe these images",
                    # Large images to activate caching
                    {
                        "type": "image",
                        "source": "https://assets.entrepreneur.com/content/3x2/2000/20200429211042-GettyImages-1164615296.jpeg",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image",
                        "source": "https://www.bigbear.com/imager/s3_us-west-1_amazonaws_com/big-bear/images/Scenic-Snow/89xVzXp1_00588cdef1e3d54756582b576359604b.jpeg",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    # Assert a cache write or cache hit
    assert (
        messages.usage.cache_creation_input_tokens > 0
        or messages.usage.cache_read_input_tokens > 0
    )


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_no_response_model(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
    system_message = (
        "You are a helpful assistant that can describe images. "
        "If looking at an image, reply with 'This is an image' and nothing else."
    )
    # Test with OpenAI style messages
    response = client.chat.completions.create(
        response_model=None,
        model=model,  # Ensure this is a vision-capable model
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": "https://pbs.twimg.com/profile_images/1816950591857233920/ZBxrWCbX_400x400.jpg",
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    assert response.content[0].text.startswith("This is an image")

    # Test with Anthropic style messages
    response = client.chat.completions.create(
        response_model=None,
        model=model,  # Ensure this is a vision-capable model
        system=system_message,
        messages=[
            {
                "role": "user",
                "content": "https://pbs.twimg.com/profile_images/1816950591857233920/ZBxrWCbX_400x400.jpg",
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    assert response.content[0].text.startswith("This is an image")

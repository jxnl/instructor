import pytest
from instructor.multimodal import Image
import instructor
from pydantic import Field, BaseModel
from itertools import product
from .util import models, modes


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description(model, mode, client):
    client = instructor.patch(client, mode=mode)

    class ImageDescription(BaseModel):
        objects: list[str] = Field(..., description="The objects in the image")
        scene: str = Field(..., description="The scene of the image")
        colors: list[str] = Field(..., description="The colors in the image")

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
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image

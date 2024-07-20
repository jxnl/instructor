from itertools import product
from typing import Literal

import pytest
import instructor
import google.generativeai as genai

from pydantic import BaseModel

from ..util import models, modes


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Literal["spam", "not_spam"]


data = [
    ("I am a spammer", "spam"),
    ("I am not a spammer", "not_spam"),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_classification(model, data, mode):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)

    input, expected = data
    resp = client.chat.completions.create(
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {input}",
            },
        ],
    )
    assert resp.class_label == expected


# Adjust the prediction model to accommodate a list of labels
class MultiClassPrediction(BaseModel):
    predicted_labels: list[Literal["billing", "general_query", "hardware"]]


data = [
    (
        "I am having trouble with my billing",
        ["billing"],
    ),
    (
        "I am having trouble with my hardware",
        ["hardware"],
    ),
    (
        "I have a general query and a billing issue",
        ["general_query", "billing"],
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_multi_classify(model, data, mode):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)

    input, expected = data

    resp = client.chat.completions.create(
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {input}",
            },
        ],
    )
    assert set(resp.predicted_labels) == set(expected)

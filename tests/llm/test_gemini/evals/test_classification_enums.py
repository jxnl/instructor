import enum
from itertools import product

import pytest
import instructor
import google.generativeai as genai

from pydantic import BaseModel

from ..util import models, modes


class Labels(str, enum.Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Labels


data = [
    (
        "I am a spammer",
        Labels.SPAM,
    ),
    (
        "I am not a spammer",
        Labels.NOT_SPAM,
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_classification(model, data, mode):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)

    input, expected = data
    resp = client.chat.completions.create(
        response_model=SinglePrediction,
        strict=False,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {input}",
            },
        ],
    )
    assert resp.class_label == expected


# Define new Enum class for multiple labels
class MultiLabels(str, enum.Enum):
    BILLING = "billing"
    GENERAL_QUERY = "general_query"
    HARDWARE = "hardware"


# Adjust the prediction model to accommodate a list of labels
class MultiClassPrediction(BaseModel):
    predicted_labels: list[MultiLabels]


data = [
    (
        "I am having trouble with my billing",
        [MultiLabels.BILLING],
    ),
    (
        "I am having trouble with my hardware",
        [MultiLabels.HARDWARE],
    ),
    (
        "I have a general query and a billing issue",
        [MultiLabels.GENERAL_QUERY, MultiLabels.BILLING],
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_multi_classify(model, data, mode):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)

    input, expected = data

    resp = client.chat.completions.create(
        response_model=MultiClassPrediction,
        strict=False,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {input}",
            },
        ],
    )
    assert set(resp.predicted_labels) == set(expected)

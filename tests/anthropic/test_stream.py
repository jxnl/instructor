from pydantic import BaseModel
from typing import List
import instructor
from instructor.dsl.partial import Partial

from anthropic import Anthropic

def test_partial_model():
    class UserExtract(BaseModel):
        name: str
        age: int

    create = instructor.patch(create=Anthropic().messages.stream, mode=instructor.function_calls.Mode.ANTHROPIC_TOOLS)
    stream = create(
        model="claude-3-opus-20240229",
        response_model=Partial[UserExtract],
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    with stream as model:
        for m in model:
            assert isinstance(m, UserExtract)

def test_partial_model_with_list():
    class User(BaseModel):
        name: str
        email: str
        twitter: str

    class MeetingInfo(BaseModel):
        users: List[User]
        date: str
        location: str
        budget: int
        deadline: str
    
    text_block = """
        In our recent online meeting, participants from various backgrounds joined to discuss the upcoming tech conference. The names and contact details of the participants were as follows:

        - Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
        - Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
        - Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

        During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024, at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher, will be our keynote speaker.

        The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities. Each participant is expected to contribute an article to the conference blog by February 20th.

        A follow-up meetingis scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
        """
    
    client = instructor.patch(create=Anthropic().messages.stream, mode=instructor.function_calls.Mode.ANTHROPIC_TOOLS)

    stream = client(
        model="claude-3-opus-20240229",
        response_model=instructor.Partial[MeetingInfo],
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Get the information about the meeting and the users {text_block}",
            },
        ],
    )
    
    with stream as model:
        for m in model:
            assert isinstance(m, MeetingInfo)

def test_partial_nested_model():
    class Address(BaseModel):
        house_number: int
        street_name: str

    class User(BaseModel):
        name: str
        age: int
        address: Address
        
    create = instructor.patch(create=Anthropic().messages.stream, mode=instructor.function_calls.Mode.ANTHROPIC_TOOLS)
    
    stream = create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old and lives at 123 First Avenue.",
            }
        ],
        response_model=Partial[User],
    ) # type: ignore
    
    with stream as model:
        for m in model:
            assert isinstance(m, User)
            if m.address:
                assert isinstance(m.address, Address)
from openai import AsyncOpenAI, OpenAI
from instructor.chat import InstructorAsyncOpenAIChat, InstructorOpenAIChat

from instructor.function_calls import Mode


class InstructorOpenAI(OpenAI):
    chat: InstructorOpenAIChat

    def __init__(self, openai_client: OpenAI, mode=Mode.FUNCTIONS) -> None:
        self.__dict__.update(openai_client.__dict__)
        self.chat = InstructorOpenAIChat(openai_client.chat, mode)


class InstructorAsyncOpenAI(AsyncOpenAI):
    chat: InstructorAsyncOpenAIChat

    def __init__(self, openai_client: OpenAI, mode=Mode.FUNCTIONS) -> None:
        self.__dict__.update(openai_client.__dict__)
        self.chat = InstructorAsyncOpenAIChat(openai_client.chat, mode)

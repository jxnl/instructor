from openai.resources import Chat, AsyncChat
from instructor.completions import (
    InstructorAsyncOpenAIChatCompletions,
    InstructorOpenAIChatCompletions,
)

from instructor.function_calls import Mode


class InstructorOpenAIChat(Chat):
    completions: InstructorOpenAIChatCompletions

    def __init__(self, openai_chat: Chat, mode: Mode):
        self.__dict__.update(openai_chat.__dict__)
        self.completions = InstructorOpenAIChatCompletions(
            openai_chat.completions, mode
        )


class InstructorAsyncOpenAIChat(AsyncChat):
    completions: InstructorAsyncOpenAIChatCompletions

    def __init__(self, openai_chat: AsyncChat, mode: Mode):
        self.__dict__.update(openai_chat.__dict__)
        self.completions = InstructorAsyncOpenAIChatCompletions(
            openai_chat.completions, mode
        )

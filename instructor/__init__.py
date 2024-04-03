from .mode import Mode
from .process_response import handle_response_model
from .distil import FinetuneFormat, Instructions
from .dsl import (
    CitationMixin,
    Maybe,
    Partial,
    IterableModel,
    llm_validator,
    openai_moderation,
)
from .function_calls import OpenAISchema, openai_schema
from .patch import apatch, patch
from .process_response import handle_parallel_model
from .client import (
    Instructor,
    AsyncInstructor,
    from_openai,
    from_litellm,
    Provider,
)


__all__ = [
    "Instructor",
    "from_openai",
    "from_litellm",
    "AsyncInstructor",
    "Provider",
    "OpenAISchema",
    "CitationMixin",
    "IterableModel",
    "Maybe",
    "Partial",
    "openai_schema",
    "Mode",
    "patch",
    "apatch",
    "llm_validator",
    "openai_moderation",
    "FinetuneFormat",
    "Instructions",
    "handle_parallel_model",
    "handle_response_model",
]

try:
    import anthropic
    from .client_anthropic import from_anthropic

    __all__.append("from_anthropic")
except ImportError:
    pass

try:
    import groq
    from .client_groq import from_groq

    __all__.append("from_groq")
except ImportError:
    pass

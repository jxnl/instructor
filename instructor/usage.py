from typing import Any, Protocol, TypeVar
from dataclasses import dataclass

T_Model = TypeVar("T_Model", bound="Response")


@dataclass
class UnifiedUsage:
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def update(self, other: "UnifiedUsage") -> None:
        self.total_tokens += other.total_tokens
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class Response(Protocol):
    usage: Any
    unified_usage: UnifiedUsage


def convert_to_unified_usage(usage: Any) -> UnifiedUsage:
    if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
        # OpenAI, Together, Groq, Mistral, Databricks, LiteLLM
        return UnifiedUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
    elif hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        # Anthropic, Cohere
        total = getattr(usage, "total_tokens", usage.input_tokens + usage.output_tokens)
        return UnifiedUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=total,
        )
    elif hasattr(usage, "prompt_token_count") and hasattr(
        usage, "candidates_token_count"
    ):
        # Gemini, VertexAI
        return UnifiedUsage(
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
        )
    else:
        raise ValueError(f"Unknown usage format: {usage}")


def extract_usage(response: Any) -> Any:
    if hasattr(response, "usage"):
        return response.usage
    elif hasattr(response, "usage_metadata"):
        return response.usage_metadata
    elif hasattr(response, "meta") and hasattr(response.meta, "billed_units"):
        return response.meta.billed_units
    raise ValueError("No usage information found")


def update_total_usage(response: Any, total_usage: UnifiedUsage) -> Any:
    try:
        response_usage = extract_usage(response)
        unified_usage = convert_to_unified_usage(response_usage)
        total_usage.update(unified_usage)
    except ValueError as e:
        print(f"Error updating usage: {e}")
    return total_usage

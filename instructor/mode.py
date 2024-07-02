import enum
import warnings


class _WarnOnFunctionsAccessEnumMeta(enum.EnumMeta):
    def __getattribute__(cls, name: str):
        if name == "FUNCTIONS":
            warnings.warn(
                "FUNCTIONS is deprecated and will be removed in future versions",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getattribute__(name)


class Mode(enum.Enum, metaclass=_WarnOnFunctionsAccessEnumMeta):
    """The mode to use for patching the client"""

    FUNCTIONS = "function_call"
    PARALLEL_TOOLS = "parallel_tool_call"
    TOOLS = "tool_call"
    MISTRAL_TOOLS = "mistral_tools"
    JSON = "json_mode"
    MD_JSON = "markdown_json_mode"
    JSON_SCHEMA = "json_schema_mode"
    ANTHROPIC_TOOLS = "anthropic_tools"
    ANTHROPIC_JSON = "anthropic_json"
    COHERE_TOOLS = "cohere_tools"
    VERTEXAI_TOOLS = "vertexai_tools"
    VERTEXAI_JSON = "vertexai_json"
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"

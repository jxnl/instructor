import enum
import warnings


class Mode(enum.Enum):
    """The mode to use for patching the client"""

    FUNCTIONS = "function_call"
    PARALLEL_TOOLS = "parallel_tool_call"
    TOOLS = "tool_call"
    STRUCTURED_OUTPUTS = "structured_output"
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
    COHERE_JSON_SCHEMA = "json_object"

    @classmethod
    def warn_mode_functions_deprecation(cls):
        warnings.warn(
            "The FUNCTIONS mode is deprecated and will be removed in future versions",
            DeprecationWarning,
            stacklevel=2,
        )

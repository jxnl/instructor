# type: ignore[all]
from typing import Any, Dict
from jinja2 import Template
from textwrap import dedent


def apply_template(text: str, context: Dict[str, Any]) -> str:
    """Apply Jinja2 template to the given text."""
    return dedent(Template(text).render(**context))


def process_message(message: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Process a single message, applying templates to its content."""
    # VertexAI Support
    if hasattr(message, "parts") and isinstance(message.parts, list):
        import vertexai.generative_models as gm

        message.parts = [
            (
                gm.Part.from_text(apply_template(part.text, context))
                if hasattr(part, "text")
                else part
            )
            for part in message.parts
        ]
        return

    # OpenAI format
    if isinstance(message.get("content"), str):
        message["content"] = apply_template(message["content"], context)
        return

    # Anthropic format
    if isinstance(message.get("content"), list):
        for part in message["content"]:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = apply_template(part["text"], context)
        return

    # Gemini Support
    if isinstance(message.get("parts"), list):
        message["parts"] = [
            apply_template(part, context) if isinstance(part, str) else part
            for part in message["parts"]
        ]
        return

    # Cohere format
    if isinstance(message.get("message"), str):
        message["message"] = apply_template(message["message"], context)


def handle_templating(
    kwargs: Dict[str, Any], context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Handle templating for messages using the provided context.

    This function processes messages, applying Jinja2 templating to their content
    using the provided context. It supports various message formats including
    OpenAI, Anthropic, Cohere, VertexAI, and Gemini.

    Args:
        kwargs (Dict[str, Any]): Keyword arguments being passed to the create method.
        context (Dict[str, Any] | None, optional): Variables to use in templating. Defaults to None.

    Returns:
        Dict[str, Any]: The processed kwargs with templated content.

    Raises:
        ValueError: If no recognized message format is found in kwargs.
    """
    if not context:
        return kwargs

    new_kwargs = kwargs.copy()

    # Handle Cohere's message field
    if "message" in new_kwargs:
        new_kwargs["message"] = apply_template(new_kwargs["message"], context)
        new_kwargs["chat_history"] = handle_templating(
            new_kwargs["chat_history"], context
        )
        return new_kwargs

    messages = new_kwargs.get("messages") or new_kwargs.get("contents")
    if not messages:
        raise ValueError("Expected 'message', 'messages' or 'contents' in kwargs")

    for message in messages:
        process_message(message, context)

    return new_kwargs

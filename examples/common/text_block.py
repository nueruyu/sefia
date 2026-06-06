from dataclasses import dataclass


@dataclass
class TextBlock:
    """A prompt argument value that should be rendered as raw text."""

    value: str

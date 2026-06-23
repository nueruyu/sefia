from dataclasses import dataclass

from .llm._client import LLMClient


@dataclass(frozen=True)
class ModelProfile:
    """
    A named, reusable bundle of model configuration for ``@infer`` functions.

    A profile pairs a ``name`` with the :class:`~sefia.llm.LLMClient` used to run
    inference for any function decorated with ``@model("<name>")``. Profiles are
    registered up front on the :class:`~sefia.Session` (``profiles=[...]``), and
    selected per function by name, so the selection at the call site is decoupled
    from the concrete client — a test can bind the same name to a mock.

    Model *settings* (temperature, max tokens, ...) are captured today by how the
    client is constructed (e.g. ``LiteLLMClient(model=..., temperature=0.2)``).
    The profile is the seam where first-class, client-agnostic settings can be
    added later without touching any call site.
    """

    name: str
    client: LLMClient

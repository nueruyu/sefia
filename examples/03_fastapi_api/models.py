"""Domain and API schemas for the FastAPI example."""

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator


@dataclass
class Brief:
    """A structured content brief produced by the Interviewer agent.

    Defined as a dataclass (like the other examples' result types) so it is the
    return type the ``@infer`` agent constructs.
    """

    topic: str
    goal: str
    audience: str


# --- Requests ---------------------------------------------------------------


class TurnRequest(BaseModel):
    """Start with an empty body, or resolve an explicitly identified request."""

    model_config = ConfigDict(extra="forbid")

    interaction_id: str | None = None
    result: JsonValue = None

    @model_validator(mode="after")
    def require_resolution_pair(self) -> Self:
        if (self.interaction_id is not None) != ("result" in self.model_fields_set):
            raise ValueError("Provide both interaction_id and result, or neither.")
        return self


# --- Responses --------------------------------------------------------------


class SessionCreatedResponse(BaseModel):
    session_id: str


class InteractionRequiredResponse(BaseModel):
    """The workflow paused to wait for input."""

    status: Literal["interaction_required"] = "interaction_required"
    interaction_id: str
    request: JsonValue


class BriefSchema(BaseModel):
    topic: str
    goal: str
    audience: str

    @classmethod
    def from_brief(cls, brief: Brief) -> "BriefSchema":
        return cls(topic=brief.topic, goal=brief.goal, audience=brief.audience)


class InterviewCompletedResponse(BaseModel):
    status: Literal["completed"] = "completed"
    brief: BriefSchema


# Discriminated union used as the FastAPI ``response_model``.
InterviewResponse = InterviewCompletedResponse | InteractionRequiredResponse

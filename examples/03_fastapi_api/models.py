"""Domain and API schemas for the FastAPI example."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


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
    """A turn for a human-in-the-loop workflow.

    ``input`` is the initial request on the first turn, or the answer to a
    pending question on later turns. ``reply_to`` targets a specific pending
    question when more than one is outstanding.
    """

    input: str
    reply_to: str | None = None


# --- Responses --------------------------------------------------------------


class SessionCreatedResponse(BaseModel):
    session_id: str


class InputRequiredResponse(BaseModel):
    """The workflow paused to wait for human input."""

    status: Literal["input_required"] = "input_required"
    interaction_id: str
    question: str


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
InterviewResponse = InterviewCompletedResponse | InputRequiredResponse

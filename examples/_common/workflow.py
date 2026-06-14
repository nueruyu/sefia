from dataclasses import dataclass, field


@dataclass
class Interaction:
    """Represents a single interaction, which could be the initial topic or a Q&A pair."""

    id: str
    answer: str | None


@dataclass
class WorkflowState:
    """Represents the state of a long-running workflow."""

    interactions: list[Interaction] = field(default_factory=list)

    @classmethod
    def from_initial_input(cls, initial_input: str) -> "WorkflowState":
        """Creates a session state from the initial input."""
        return cls(interactions=[Interaction(id="__initial__", answer=initial_input)])

    @property
    def initial_input(self) -> str:
        """The initial input that started the session."""
        if not self.interactions or self.interactions[0].answer is None:
            raise RuntimeError("WorkflowState has no initial input.")
        return self.interactions[0].answer

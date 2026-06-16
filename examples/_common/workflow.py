from dataclasses import dataclass


@dataclass
class WorkflowState:
    """Represents the state of a long-running workflow."""

    initial_input: str | None = None

    @classmethod
    def from_initial_input(cls, initial_input: str) -> "WorkflowState":
        """Creates a workflow state from the initial input."""
        return cls(initial_input=initial_input)

    @property
    def has_initial_input(self) -> bool:
        return self.initial_input is not None

    def require_initial_input(self) -> str:
        """Returns the initial input, or raises when it has not been initialized."""
        if self.initial_input is None:
            raise RuntimeError("WorkflowState has no initial input.")
        return self.initial_input

import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Interaction:
    """Represents a single interaction, which could be the initial topic or a Q&A pair."""

    id: str
    answer: str | None


@dataclass
class WorkflowState:
    """Represents the state of a long-running workflow."""

    interactions: list[Interaction] = field(default_factory=list)
    _interactions_by_id: dict[str, Interaction] = field(init=False, repr=False)

    @classmethod
    def from_initial_input(cls, initial_input: str) -> "WorkflowState":
        """Creates a session state from the initial input."""
        return cls(interactions=[Interaction(id="__initial__", answer=initial_input)])

    def __post_init__(self):
        self._interactions_by_id = {i.id: i for i in self.interactions}

    @property
    def initial_input(self) -> str:
        """The initial input that started the session."""
        if not self.interactions:
            raise RuntimeError("WorkflowState has no initial input.")
        initial_input = self.interactions[0].answer
        if initial_input is None:
            raise RuntimeError("Initial input cannot be pending.")
        return initial_input

    @property
    def pending_interaction(self) -> Interaction | None:
        """The last interaction that is still awaiting an answer."""
        if self.interactions and self.interactions[-1].answer is None:
            return self.interactions[-1]
        return None

    def add_pending_interaction(self, interaction_id: str) -> None:
        """Adds a new interaction that is awaiting an answer."""
        if self.pending_interaction:
            raise RuntimeError(
                "Cannot add a new pending interaction while another one is active."
            )
        interaction = Interaction(id=interaction_id, answer=None)
        self.interactions.append(interaction)
        self._interactions_by_id[interaction_id] = interaction

    def update_pending_answer(self, answer: str) -> None:
        """Sets the answer for the current pending interaction."""
        pending = self.pending_interaction
        if not pending:
            raise RuntimeError("No pending interaction to update.")
        pending.answer = answer

    def get_answer_by_id(self, interaction_id: str) -> str | None:
        """Gets an answer for a completed interaction by its ID."""
        interaction = self._interactions_by_id.get(interaction_id)
        return interaction.answer if interaction else None


class Manager:
    """Manages the lifecycle of user chat sessions, including the active session ID."""

    def __init__(self, session_dir: Path):
        self._session_dir = session_dir
        self._active_session_file = self._session_dir / "active_session.txt"
        self._session_dir.mkdir(exist_ok=True)

    def get_active_session_id(self) -> str | None:
        """Gets the ID of the currently active session, if one exists."""
        if self._active_session_file.exists():
            return self._active_session_file.read_text(encoding="utf-8").strip()
        return None

    def set_active_session_id(self, session_id: str) -> None:
        """Sets the active session ID."""
        self._active_session_file.write_text(session_id, encoding="utf-8")

    def create_new_session_id(self) -> str:
        """Generates a new unique session ID."""
        return str(uuid.uuid4())

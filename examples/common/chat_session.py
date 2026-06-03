from dataclasses import dataclass, field


@dataclass
class Interaction:
    """Represents a single interaction, which could be the initial topic or a Q&A pair."""

    id: str
    answer: str | None


@dataclass
class ChatSessionState:
    """Represents the state of a long-running chat session."""

    _interactions: list[Interaction] = field(default_factory=list)
    _interactions_by_id: dict[str, Interaction] = field(init=False, repr=False)

    @classmethod
    def from_initial_topic(cls, initial_topic: str) -> "ChatSessionState":
        """Creates a session state from the initial topic."""
        return cls(_interactions=[Interaction(id="__initial__", answer=initial_topic)])

    def __post_init__(self):
        self._interactions_by_id = {i.id: i for i in self._interactions}

    @property
    def initial_topic(self) -> str:
        """The initial topic that started the session."""
        if not self._interactions:
            raise RuntimeError("ChatSessionState has no initial topic.")
        initial_topic = self._interactions[0].answer
        if initial_topic is None:
            raise RuntimeError("Initial topic cannot be pending.")
        return initial_topic

    @property
    def pending_interaction(self) -> Interaction | None:
        """The last interaction that is still awaiting an answer."""
        if self._interactions and self._interactions[-1].answer is None:
            return self._interactions[-1]
        return None

    def add_pending_interaction(self, interaction_id: str) -> None:
        """Adds a new interaction that is awaiting an answer."""
        if self.pending_interaction:
            raise RuntimeError(
                "Cannot add a new pending interaction while another one is active."
            )
        interaction = Interaction(id=interaction_id, answer=None)
        self._interactions.append(interaction)
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

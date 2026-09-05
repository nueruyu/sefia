import uuid
from dataclasses import dataclass, field
from typing import Callable

import pytest
from glyff import Domain

from sefia import Tools
from sefia.exceptions import PauseException
from sefia.llm import LLMCompletion
from sefia.testing import MockLLMClient, result_completion, tool_calls_completion
from sefios import (
    MemoryPersistence,
    SessionScope,
    domain,
    get_call_state_store,
    get_state,
    state,
)

infer = domain("packages.sefios.tests.scenarios.test_stateful_tools", version="1").infer


@dataclass
class Report:
    topic: str
    summary: str
    sources: list[str]


# --- State models for testing ---
@dataclass
class _InputCallState:
    interaction_id: str | None = None


@dataclass
class Answer:
    content: str


@state(key="interaction_state")
@dataclass
class InteractionState:
    answers: dict[str, Answer] = field(default_factory=lambda: {})


# --- Test tool with internal state management ---
@dataclass
class Input:
    def __init__(self, on_interrupt: Callable[[str, str], None] | None = None):
        self._on_interrupt = on_interrupt

    @Domain("sefios.tests", version="1").engrave
    async def ask_user(self, prompt: str) -> str:
        call_store = get_call_state_store("internal_state", _InputCallState)
        call_state = await call_store.ensure()

        try:
            if call_state.interaction_id is None:
                call_state.interaction_id = str(uuid.uuid4())
                await call_store.save(call_state)
                raise PauseException()

            session_store = get_state().get(InteractionState)
            interaction_state = await session_store.get()

            if (
                interaction_state is None
                or call_state.interaction_id not in interaction_state.answers
            ):
                raise PauseException()

            answer = interaction_state.answers.pop(call_state.interaction_id)
            await session_store.save(interaction_state)
            return answer.content
        except PauseException:
            if self._on_interrupt and call_state.interaction_id:
                self._on_interrupt(call_state.interaction_id, prompt)
            raise


class TestStatefulTool:
    async def test_ask_user_interrupts_and_resumes_correctly(
        self,
        make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
    ) -> None:
        mock_responses = [
            tool_calls_completion(("Input_ask_user", {"prompt": "What is your name?"})),
            result_completion(
                Report(
                    topic="User Info",
                    summary="The user's name is Alice.",
                    sources=[],
                )
            ),
        ]
        mock_llm = make_mock_llm(mock_responses)
        interrupt_details: dict[str, str] = {}

        def on_interrupt(interaction_id: str, prompt: str) -> None:
            interrupt_details["id"] = interaction_id
            interrupt_details["prompt"] = prompt

        @dataclass
        class Agent:
            _tool: Tools[Input]

            def __init__(self, tool: Input):
                self._tool = tool

            @infer
            async def get_user_name(self) -> Report:
                """Ask the user for their name and create a report."""
                ...

        agent = Agent(Input(on_interrupt=on_interrupt))
        session_id = "stateful-tool-test-1"
        persistence = MemoryPersistence()

        with pytest.raises(PauseException):
            async with SessionScope(
                llm_client=mock_llm, persistence=persistence
            ).session(session_id=session_id):
                await agent.get_user_name()

        assert "id" in interrupt_details
        assert interrupt_details["prompt"] == "What is your name?"

        async with SessionScope(llm_client=mock_llm, persistence=persistence).session(
            session_id=session_id
        ):
            interaction_store = get_state().get(InteractionState)
            interaction_state = await interaction_store.ensure()
            interaction_state.answers[interrupt_details["id"]] = Answer(content="Alice")
            await interaction_store.save(interaction_state)

            report = await agent.get_user_name()

        assert report.summary == "The user's name is Alice."
        assert len(mock_llm.requests) == 2

    async def test_multiple_ask_user_calls_are_independent(
        self,
        make_mock_llm: Callable[[list[LLMCompletion]], MockLLMClient],
    ) -> None:
        mock_responses = [
            tool_calls_completion(("Input_ask_user", {"prompt": "Name?"})),
            tool_calls_completion(("Input_ask_user", {"prompt": "Age?"})),
            result_completion(
                Report(topic="Profile", summary="Alice is 99.", sources=[])
            ),
        ]
        mock_llm = make_mock_llm(mock_responses)
        interrupts: dict[str, str] = {}

        def on_interrupt(interaction_id: str, prompt: str) -> None:
            interrupts[prompt] = interaction_id

        @dataclass
        class Agent:
            _tool: Tools[Input]

            def __init__(self, tool: Input):
                self._tool = tool

            @infer
            async def get_profile(self) -> Report:
                """Ask for name, then age, then report."""
                ...

        agent = Agent(Input(on_interrupt=on_interrupt))
        session_id = "stateful-tool-test-2"
        persistence = MemoryPersistence()

        with pytest.raises(PauseException):
            async with SessionScope(
                llm_client=mock_llm, persistence=persistence
            ).session(session_id=session_id):
                await agent.get_profile()
        assert "Name?" in interrupts

        with pytest.raises(PauseException):
            async with SessionScope(
                llm_client=mock_llm, persistence=persistence
            ).session(session_id=session_id):
                interaction_store = get_state().get(InteractionState)
                interaction_state = await interaction_store.ensure()
                interaction_state.answers[interrupts["Name?"]] = Answer(content="Alice")
                await interaction_store.save(interaction_state)
                await agent.get_profile()
        assert "Age?" in interrupts

        async with SessionScope(llm_client=mock_llm, persistence=persistence).session(
            session_id=session_id
        ):
            interaction_store = get_state().get(InteractionState)
            interaction_state = await interaction_store.ensure()
            interaction_state.answers[interrupts["Age?"]] = Answer(content="99")
            await interaction_store.save(interaction_state)
            report = await agent.get_profile()

        assert report.summary == "Alice is 99."
        assert len(mock_llm.requests) == 3

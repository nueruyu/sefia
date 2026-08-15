import json
import uuid
from dataclasses import dataclass, field
from typing import Callable

import glyff
import pytest
from glyff import ArgumentCanonicalizer, Domain, Serializer
from glyff.store import MemoryBackend

from sefia import Session, Tools
from sefia.exceptions import PauseException
from sefia.llm import LLMResponse
from sefia.testing import MockLLMClient, result_response, tool_calls_response
from sefios import domain, MemorySessionStorage, get_call_state_store
from sefios._session_state import bind_session_storage, get_state_store

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

            session_store = get_state_store("interaction_state", InteractionState)
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
        serializer: Serializer,
        hasher: ArgumentCanonicalizer,
        make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
    ) -> None:
        mock_responses = [
            tool_calls_response(("Input_ask_user", {"prompt": "What is your name?"})),
            result_response(
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
        glyff_store = MemoryBackend()
        sefia_store = MemorySessionStorage(serializer=serializer)

        # --- First run: Should interrupt ---
        with pytest.raises(PauseException):
            async with glyff.Session(
                id=glyff.SessionId(session_id),
                backend=glyff_store,
                serializer=serializer,
                argument_canonicalizer=hasher,
            ) as gs:
                with bind_session_storage(sefia_store):
                    async with Session(llm_client=mock_llm, glyff_session=gs):
                        await agent.get_user_name()

        assert "id" in interrupt_details
        assert interrupt_details["prompt"] == "What is your name?"

        # --- Second run (with answer): Should succeed ---
        async with glyff.Session(
            id=glyff.SessionId(session_id),
            backend=glyff_store,
            serializer=serializer,
            argument_canonicalizer=hasher,
        ) as gs:
            with bind_session_storage(sefia_store):
                async with Session(llm_client=mock_llm, glyff_session=gs):
                    interaction_store = get_state_store(
                        "interaction_state", InteractionState
                    )
                    state = await interaction_store.ensure()
                    state.answers[interrupt_details["id"]] = Answer(content="Alice")
                    await interaction_store.save(state)

                    report = await agent.get_user_name()

        assert report.summary == "The user's name is Alice."
        assert len(mock_llm.requests) == 2
        final_messages = mock_llm.requests[1]["messages"]
        assert len(final_messages) == 4
        assert json.loads(final_messages[3]["content"]) == "Alice"

    async def test_multiple_ask_user_calls_are_independent(
        self,
        serializer: Serializer,
        hasher: ArgumentCanonicalizer,
        make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
    ) -> None:
        mock_responses = [
            tool_calls_response(("Input_ask_user", {"prompt": "Name?"})),
            tool_calls_response(("Input_ask_user", {"prompt": "Age?"})),
            result_response(
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
        glyff_store = MemoryBackend()
        sefia_store = MemorySessionStorage(serializer=serializer)

        # --- 1. Ask for name, should interrupt ---
        with pytest.raises(PauseException):
            async with glyff.Session(
                id=glyff.SessionId(session_id),
                backend=glyff_store,
                serializer=serializer,
                argument_canonicalizer=hasher,
            ) as gs:
                with bind_session_storage(sefia_store):
                    async with Session(llm_client=mock_llm, glyff_session=gs):
                        await agent.get_profile()
        assert "Name?" in interrupts

        # --- 2. Provide name, ask for age, should interrupt again ---
        with pytest.raises(PauseException):
            async with glyff.Session(
                id=glyff.SessionId(session_id),
                backend=glyff_store,
                serializer=serializer,
                argument_canonicalizer=hasher,
            ) as gs:
                with bind_session_storage(sefia_store):
                    async with Session(llm_client=mock_llm, glyff_session=gs):
                        interaction_store = get_state_store(
                            "interaction_state", InteractionState
                        )
                        state = await interaction_store.ensure()
                        state.answers[interrupts["Name?"]] = Answer(content="Alice")
                        await interaction_store.save(state)
                        await agent.get_profile()
        assert "Age?" in interrupts

        # --- 3. Provide age, should complete ---
        async with glyff.Session(
            id=glyff.SessionId(session_id),
            backend=glyff_store,
            serializer=serializer,
            argument_canonicalizer=hasher,
        ) as gs:
            with bind_session_storage(sefia_store):
                async with Session(llm_client=mock_llm, glyff_session=gs):
                    interaction_store = get_state_store(
                        "interaction_state", InteractionState
                    )
                    state = await interaction_store.ensure()
                    state.answers[interrupts["Age?"]] = Answer(content="99")
                    await interaction_store.save(state)
                    report = await agent.get_profile()

        assert report.summary == "Alice is 99."
        assert len(mock_llm.requests) == 3

import json
import uuid
from dataclasses import dataclass, field
from typing import Callable

import glyff
import pytest
from glyff import engrave
from glyff.exceptions import YieldException
from glyff.interfaces import ArgsHasher, Serializer
from glyff.stores import MemoryClient
from glyff.stores import MemorySessionStore as GlyffMemoryStore

from sefia import (
    Session,
    get_context,
    infer,
    tool,
)
from sefia.llm import LLMResponse
from sefia.stores import MemorySessionStore as SefiaMemoryStore

from ..conftest import MockLLMClient, Report


# --- State models for testing ---
@dataclass
class _AskUserState:
    interaction_id: str | None = None


@dataclass
class Answer:
    content: str


@dataclass
class InteractionState:
    answers: dict[str, Answer] = field(default_factory=dict)


# --- Test tool with internal state management ---
@dataclass
class HumanInputTool:
    def __init__(self, on_interrupt: Callable[[str, str], None] | None = None):
        self._on_interrupt = on_interrupt

    @tool
    @engrave
    async def ask_user(self, question: str) -> str:
        ctx = get_context()
        call_store = ctx.get_call_state_store("internal_state", _AskUserState)
        call_state = await call_store.ensure()

        try:
            if call_state.interaction_id is None:
                call_state.interaction_id = str(uuid.uuid4())
                await call_store.save(call_state)
                raise YieldException()

            session_store = ctx.get_state_store("interaction_state", InteractionState)
            interaction_state = await session_store.get()

            if (
                interaction_state is None
                or call_state.interaction_id not in interaction_state.answers
            ):
                raise YieldException()

            answer = interaction_state.answers.pop(call_state.interaction_id)
            await session_store.save(interaction_state)
            return answer.content
        except YieldException:
            if self._on_interrupt and call_state.interaction_id:
                self._on_interrupt(call_state.interaction_id, question)
            raise


class TestStatefulTool:
    async def test_ask_user_interrupts_and_resumes_correctly(
        self, serializer: Serializer, hasher: ArgsHasher
    ):
        mock_responses = [
            LLMResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "HumanInputTool_ask_user",
                                "arguments": {"question": "What is your name?"},
                            }
                        ]
                    }
                )
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "final_answer": {
                            "topic": "User Info",
                            "summary": "The user's name is Alice.",
                            "sources": [],
                        }
                    }
                )
            ),
        ]
        mock_llm = MockLLMClient(responses=mock_responses)
        interrupt_details = {}

        def on_interrupt(interaction_id, question):
            interrupt_details["id"] = interaction_id
            interrupt_details["question"] = question

        @dataclass
        class Agent:
            def __init__(self, tool: HumanInputTool):
                self._tool = tool

            @infer
            async def get_user_name(self) -> Report:
                """Ask the user for their name and create a report."""
                ...

        agent = Agent(HumanInputTool(on_interrupt=on_interrupt))
        session_id = "stateful-tool-test-1"

        client = MemoryClient()
        glyff_store = GlyffMemoryStore(client=client, serializer=serializer)
        sefia_store = SefiaMemoryStore(client=client, serializer=serializer)

        # --- First run: Should interrupt ---
        with pytest.raises(YieldException):
            async with glyff.Session(
                id=session_id, store=glyff_store, hasher=hasher
            ) as gs:
                async with Session(
                    llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
                ) as sefia_session:
                    await agent.get_user_name()

        assert "id" in interrupt_details
        assert interrupt_details["question"] == "What is your name?"

        # --- Second run (with answer): Should succeed ---
        async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
            async with Session(
                llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
            ) as sefia_session:
                interaction_store = sefia_session.get_state_store(
                    "interaction_state", InteractionState
                )
                state = await interaction_store.ensure()
                state.answers[interrupt_details["id"]] = Answer(content="Alice")
                await interaction_store.save(state)

                report = await agent.get_user_name()

        assert report.summary == "The user's name is Alice."
        assert len(mock_llm.requests) == 2
        final_messages = mock_llm.requests[1]["messages"]
        # system, user, assistant(tool_call), tool(result)
        assert len(final_messages) == 4
        assert json.loads(final_messages[3]["content"]) == "Alice"

    async def test_multiple_ask_user_calls_are_independent(
        self, serializer: Serializer, hasher: ArgsHasher
    ):
        mock_responses = [
            LLMResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "HumanInputTool_ask_user",
                                "arguments": {"question": "Name?"},
                            }
                        ]
                    }
                )
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "HumanInputTool_ask_user",
                                "arguments": {"question": "Age?"},
                            }
                        ]
                    }
                )
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "final_answer": {
                            "topic": "Profile",
                            "summary": "Alice is 99.",
                            "sources": [],
                        }
                    }
                )
            ),
        ]
        mock_llm = MockLLMClient(responses=mock_responses)
        interrupts = {}

        def on_interrupt(interaction_id, question):
            interrupts[question] = interaction_id

        @dataclass
        class Agent:
            def __init__(self, tool: HumanInputTool):
                self._tool = tool

            @infer
            async def get_profile(self) -> Report:
                """Ask for name, then age, then report."""
                ...

        agent = Agent(HumanInputTool(on_interrupt=on_interrupt))
        session_id = "stateful-tool-test-2"

        client = MemoryClient()
        glyff_store = GlyffMemoryStore(client=client, serializer=serializer)
        sefia_store = SefiaMemoryStore(client=client, serializer=serializer)

        # --- 1. Ask for name, should interrupt ---
        with pytest.raises(YieldException):
            async with glyff.Session(
                id=session_id, store=glyff_store, hasher=hasher
            ) as gs:
                async with Session(
                    llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
                ):
                    await agent.get_profile()
        assert "Name?" in interrupts

        # --- 2. Provide name, ask for age, should interrupt again ---
        with pytest.raises(YieldException):
            async with glyff.Session(
                id=session_id, store=glyff_store, hasher=hasher
            ) as gs:
                async with Session(
                    llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
                ) as sefia_session:
                    interaction_store = sefia_session.get_state_store(
                        "interaction_state", InteractionState
                    )
                    state = await interaction_store.ensure()
                    state.answers[interrupts["Name?"]] = Answer(content="Alice")
                    await interaction_store.save(state)
                    await agent.get_profile()
        assert "Age?" in interrupts

        # --- 3. Provide age, should complete ---
        async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
            async with Session(
                llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
            ) as sefia_session:
                interaction_store = sefia_session.get_state_store(
                    "interaction_state", InteractionState
                )
                state = await interaction_store.ensure()
                state.answers[interrupts["Age?"]] = Answer(content="99")
                await interaction_store.save(state)
                report = await agent.get_profile()

        assert report.summary == "Alice is 99."
        assert len(mock_llm.requests) == 3

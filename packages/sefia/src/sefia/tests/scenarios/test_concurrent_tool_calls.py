import asyncio
import json

import glyff
import pytest
from glyff import ArgsHasher, Serializer, engrave
from glyff.store import MemoryBackend

from sefia import Session, concurrent, infer
from sefia.exceptions import PauseException
from sefia.llm import LLMResponse

from ..conftest import MockLLMClient, Report


class HandshakeToolkit:
    """Two tools that can only both finish if the batch overlaps them."""

    def __init__(self):
        self._peer_ran = asyncio.Event()

    @concurrent
    async def wait_for_peer(self) -> str:
        """Wait until the peer tool has run."""
        await asyncio.wait_for(self._peer_ran.wait(), timeout=5)
        return "waited"

    @concurrent
    async def release_peer(self) -> str:
        """Unblock the waiting peer tool."""
        self._peer_ran.set()
        return "released"


class Researcher:
    def __init__(self, kit: HandshakeToolkit):
        self._kit = kit

    @infer
    async def generate_report(self, topic: str) -> Report:
        """Generate a report on the given topic."""
        ...


async def test_concurrent_calls_in_one_decision_overlap(
    serializer: Serializer, hasher: ArgsHasher
):
    mock_responses = [
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {"name": "HandshakeToolkit_wait_for_peer", "arguments": {}},
                        {"name": "HandshakeToolkit_release_peer", "arguments": {}},
                    ],
                }
            )
        ),
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "result",
                    "result": {"topic": "t", "summary": "s", "sources": []},
                }
            )
        ),
    ]
    mock_llm = MockLLMClient(responses=mock_responses)

    async with glyff.Session(
        id="concurrent-overlap",
        backend=MemoryBackend(),
        serializer=serializer,
        hasher=hasher,
    ) as gs:
        async with Session(llm_client=mock_llm, glyff_session=gs):
            report = await Researcher(HandshakeToolkit()).generate_report(topic="t")

    assert report.summary == "s"
    # Both results reach the model in request order: the waiting tool first,
    # even though it finished after the releasing one.
    final_messages = mock_llm.requests[1]["messages"]
    tool_messages = [m for m in final_messages if m.get("role") == "tool"]
    assert [json.loads(m["content"]) for m in tool_messages] == ["waited", "released"]


class PausingToolkit:
    """An engraved data fetch next to a human-gate that pauses the run."""

    def __init__(self):
        self.fetch_runs = 0
        self.answer: str | None = None

    @concurrent
    @engrave
    async def fetch_data(self, key: str) -> str:
        """Fetch data for a key."""
        self.fetch_runs += 1
        return f"data:{key}"

    @concurrent
    async def ask_user(self, question: str) -> str:
        """Ask the user a question."""
        if self.answer is None:
            raise PauseException("waiting for an answer")
        return self.answer


class Assistant:
    def __init__(self, kit: PausingToolkit):
        self._kit = kit

    @infer
    async def prepare_report(self, topic: str) -> Report:
        """Prepare a report on the given topic."""
        ...


async def test_pause_in_concurrent_batch_resumes_without_rerunning_sibling(
    serializer: Serializer, hasher: ArgsHasher
):
    mock_responses = [
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "PausingToolkit_fetch_data",
                            "arguments": {"key": "alpha"},
                        },
                        {
                            "name": "PausingToolkit_ask_user",
                            "arguments": {"question": "Proceed?"},
                        },
                    ],
                }
            )
        ),
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "result",
                    "result": {"topic": "t", "summary": "approved", "sources": []},
                }
            )
        ),
    ]
    mock_llm = MockLLMClient(responses=mock_responses)
    toolkit = PausingToolkit()
    assistant = Assistant(toolkit)
    glyff_store = MemoryBackend()

    # First run: the engraved sibling completes, then the batch pauses.
    with pytest.raises(PauseException):
        async with glyff.Session(
            id="concurrent-pause",
            backend=glyff_store,
            serializer=serializer,
            hasher=hasher,
        ) as gs:
            async with Session(llm_client=mock_llm, glyff_session=gs):
                await assistant.prepare_report(topic="t")

    assert toolkit.fetch_runs == 1

    # Second run: the decision and the engraved sibling replay (no re-run,
    # no extra LLM call); only the pausing tool executes again.
    toolkit.answer = "yes"
    async with glyff.Session(
        id="concurrent-pause",
        backend=glyff_store,
        serializer=serializer,
        hasher=hasher,
    ) as gs:
        async with Session(llm_client=mock_llm, glyff_session=gs):
            report = await assistant.prepare_report(topic="t")

    assert report.summary == "approved"
    assert toolkit.fetch_runs == 1
    assert len(mock_llm.requests) == 2
    final_messages = mock_llm.requests[1]["messages"]
    tool_messages = [m for m in final_messages if m.get("role") == "tool"]
    assert [json.loads(m["content"]) for m in tool_messages] == ["data:alpha", "yes"]

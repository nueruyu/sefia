import glyff
import sefia
import asyncio
import json
from dataclasses import dataclass

import pytest
from glyff import Domain
from glyff.store import MemoryBackend

from sefia import Tools, concurrent
from sefia.exceptions import PauseException
from sefia.testing import (
    MockLLMClient,
    memory_session,
    result_response,
    tool_calls_response,
)


@dataclass
class Report:
    topic: str
    summary: str
    sources: list[str]


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
    _kit: Tools[HandshakeToolkit]

    def __init__(self, kit: HandshakeToolkit):
        self._kit = kit

    @sefia.Domain(
        glyff.Domain(
            "packages.sefia.tests.scenarios.test_concurrent_tool_calls", version="1"
        )
    ).infer(name="Researcher.generate_report")
    async def generate_report(self, topic: str) -> Report:
        """Generate a report on the given topic."""
        ...


async def test_concurrent_calls_in_one_decision_overlap():
    mock_llm = MockLLMClient(
        responses=[
            tool_calls_response(
                ("HandshakeToolkit_wait_for_peer", {}),
                ("HandshakeToolkit_release_peer", {}),
            ),
            result_response(Report(topic="t", summary="s", sources=[])),
        ]
    )

    async with memory_session(mock_llm, session_id="concurrent-overlap"):
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
    @Domain("sefia.tests", version="1").engrave
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
    _kit: Tools[PausingToolkit]

    def __init__(self, kit: PausingToolkit):
        self._kit = kit

    @sefia.Domain(
        glyff.Domain(
            "packages.sefia.tests.scenarios.test_concurrent_tool_calls", version="1"
        )
    ).infer(name="Assistant.prepare_report")
    async def prepare_report(self, topic: str) -> Report:
        """Prepare a report on the given topic."""
        ...


async def test_pause_in_concurrent_batch_resumes_without_rerunning_sibling():
    mock_llm = MockLLMClient(
        responses=[
            tool_calls_response(
                ("PausingToolkit_fetch_data", {"key": "alpha"}),
                ("PausingToolkit_ask_user", {"question": "Proceed?"}),
            ),
            result_response(Report(topic="t", summary="approved", sources=[])),
        ]
    )
    toolkit = PausingToolkit()
    assistant = Assistant(toolkit)
    # The two runs share one backend, so the second replays the first's steps.
    glyff_store = MemoryBackend()

    # First run: the engraved sibling completes, then the batch pauses.
    with pytest.raises(PauseException):
        async with memory_session(
            mock_llm, session_id="concurrent-pause", backend=glyff_store
        ):
            await assistant.prepare_report(topic="t")

    assert toolkit.fetch_runs == 1

    # Second run: the decision and the engraved sibling replay (no re-run,
    # no extra LLM call); only the pausing tool executes again.
    toolkit.answer = "yes"
    async with memory_session(
        mock_llm, session_id="concurrent-pause", backend=glyff_store
    ):
        report = await assistant.prepare_report(topic="t")

    assert report.summary == "approved"
    assert toolkit.fetch_runs == 1
    assert len(mock_llm.requests) == 2
    final_messages = mock_llm.requests[1]["messages"]
    tool_messages = [m for m in final_messages if m.get("role") == "tool"]
    assert [json.loads(m["content"]) for m in tool_messages] == ["data:alpha", "yes"]

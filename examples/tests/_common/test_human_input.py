import pytest
from sefios.tools import HumanInputRequest, HumanInputResult

from examples._common.human_input import (
    AmbiguousHumanInputError,
    CLIHumanInputAdapter,
    CLIHumanInputReceiver,
    HumanInputSessionStore,
    UnknownHumanInputError,
)


def _request(interaction_id: str) -> dict:
    return {"id": interaction_id, "question": f"question for {interaction_id}?"}


class TestHumanInputSessionStore:
    @pytest.fixture
    def store(self, session_store):
        human_store = HumanInputSessionStore()
        with human_store.use_session_store(session_store):
            yield human_store

    async def test_requires_bound_session(self):
        store = HumanInputSessionStore()

        with pytest.raises(RuntimeError):
            await store.pending_requests()

    async def test_pending_requests_empty_by_default(self, store):
        assert await store.pending_requests() == {}

    async def test_save_and_read_pending(self, store):
        await store.save_pending_requests({"a": _request("a")})

        assert await store.pending_requests() == {"a": _request("a")}

    async def test_answered_requests_are_dropped_from_pending(self, store):
        await store.save_pending_requests({"a": _request("a"), "b": _request("b")})
        await store.set_answer("a", "answered")

        pending = await store.pending_requests()

        assert set(pending) == {"b"}

    async def test_saving_empty_pending_clears_state(self, store):
        await store.save_pending_requests({"a": _request("a")})
        await store.save_pending_requests({})

        assert await store.pending_requests() == {}

    async def test_get_answer_missing(self, store):
        assert await store.get_answer("missing") is None

    async def test_queue_inputs_are_returned_in_order(self, store):
        await store.queue_input("first")
        await store.queue_input("second")

        assert await store.pop_queued_input() == "first"
        assert await store.pop_queued_input() == "second"
        assert await store.pop_queued_input() is None


class TestCLIHumanInputReceiver:
    @pytest.fixture
    def store(self, session_store):
        human_store = HumanInputSessionStore()
        with human_store.use_session_store(session_store):
            yield human_store

    async def test_input_is_queued_when_nothing_pending(self, store):
        receiver = CLIHumanInputReceiver(store)

        await receiver.receive_input("hello")

        assert await store.pop_queued_input() == "hello"

    async def test_single_pending_request_is_answered(self, store):
        await store.save_pending_requests({"only": _request("only")})
        receiver = CLIHumanInputReceiver(store)

        await receiver.receive_input("the answer")

        assert await store.get_answer("only") == "the answer"

    async def test_multiple_pending_requires_reply_to(self, store):
        await store.save_pending_requests({"a": _request("a"), "b": _request("b")})
        receiver = CLIHumanInputReceiver(store)

        with pytest.raises(AmbiguousHumanInputError) as exc_info:
            await receiver.receive_input("ambiguous")

        assert sorted(exc_info.value.interaction_ids) == ["a", "b"]

    async def test_reply_to_targets_specific_request(self, store):
        await store.save_pending_requests({"a": _request("a"), "b": _request("b")})
        receiver = CLIHumanInputReceiver(store)

        await receiver.receive_input("for b", reply_to="b")

        assert await store.get_answer("b") == "for b"
        assert await store.get_answer("a") is None

    async def test_reply_to_unknown_request_raises(self, store):
        await store.save_pending_requests({"a": _request("a")})
        receiver = CLIHumanInputReceiver(store)

        with pytest.raises(UnknownHumanInputError) as exc_info:
            await receiver.receive_input("oops", reply_to="missing")

        assert exc_info.value.interaction_id == "missing"


class TestCLIHumanInputAdapter:
    @pytest.fixture
    def adapter(self, session_store):
        adapter = CLIHumanInputAdapter()
        with adapter.store.use_session_store(session_store):
            yield adapter

    def test_create_tool_returns_human_input_tool(self, adapter):
        from sefios.tools import HumanInputTool

        assert isinstance(adapter.create_tool(), HumanInputTool)

    async def test_handle_request_records_pending_and_notifies(self, session_store):
        seen: list[HumanInputRequest] = []
        adapter = CLIHumanInputAdapter(on_request=seen.append)

        with adapter.store.use_session_store(session_store):
            request = HumanInputRequest(interaction_id="x", question="why?")
            await adapter._handle_request(request)

            pending = await adapter.store.pending_requests()

        assert "x" in pending
        assert seen == [request]

    async def test_handle_complete_removes_pending(self, adapter):
        await adapter.store.save_pending_requests({"x": _request("x")})

        await adapter._handle_complete(
            HumanInputResult(interaction_id="x", question="why?", answer="because")
        )

        assert await adapter.store.pending_requests() == {}

    async def test_get_answer_returns_stored_answer(self, adapter):
        await adapter.store.set_answer("x", "stored")

        request = HumanInputRequest(interaction_id="x", question="why?")
        assert await adapter._get_answer(request) == "stored"

    async def test_get_answer_consumes_queued_input_when_unblocked(self, adapter):
        await adapter.store.queue_input("queued")

        request = HumanInputRequest(interaction_id="x", question="why?")
        assert await adapter._get_answer(request) == "queued"

    async def test_get_answer_does_not_consume_queue_with_other_pending(self, adapter):
        await adapter.store.save_pending_requests({"other": _request("other")})
        await adapter.store.queue_input("queued")

        request = HumanInputRequest(interaction_id="x", question="why?")

        assert await adapter._get_answer(request) is None
        # The queued input is left untouched for later.
        assert await adapter.store.pop_queued_input() == "queued"

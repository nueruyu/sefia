import pytest
from sefia_typer import (
    AmbiguousInputError,
    InputCoordinator,
    InputReceiver,
    InputRequest,
    InputStore,
    UnknownInputError,
)
from sefia_typer._input import _to_input_text


def _request(interaction_id: str) -> dict:
    return {"id": interaction_id, "prompt": f"prompt for {interaction_id}?"}


class TestToInputText:
    def test_plain_string_is_stripped(self):
        assert _to_input_text("  hello  ") == "hello"

    def test_list_is_joined_with_spaces(self):
        assert _to_input_text(["hello", "world"]) == "hello world"

    def test_list_result_is_stripped(self):
        assert _to_input_text(["  hello  "]) == "hello"

    def test_empty_list_is_empty_string(self):
        assert _to_input_text([]) == ""


class TestInputStore:
    @pytest.fixture
    def store(self, kv_store):
        human_store = InputStore()
        with human_store.use_store(kv_store):
            yield human_store

    async def test_requires_bound_store(self):
        store = InputStore()

        with pytest.raises(RuntimeError):
            await store.pending_requests()

    async def test_pending_requests_empty_by_default(self, store):
        assert await store.pending_requests() == {}

    async def test_save_and_read_pending(self, store):
        await store.save_pending_requests({"a": _request("a")})

        assert await store.pending_requests() == {"a": _request("a")}

    async def test_answered_requests_are_dropped_from_pending(self, store):
        await store.save_pending_requests({"a": _request("a"), "b": _request("b")})
        await store.set_input("a", "answered")

        pending = await store.pending_requests()

        assert set(pending) == {"b"}

    async def test_saving_empty_pending_clears_state(self, store):
        await store.save_pending_requests({"a": _request("a")})
        await store.save_pending_requests({})

        assert await store.pending_requests() == {}

    async def test_get_input_missing(self, store):
        assert await store.get_input("missing") is None

    async def test_queue_inputs_are_returned_in_order(self, store):
        await store.queue_input("first")
        await store.queue_input("second")

        assert await store.pop_queued_input() == "first"
        assert await store.pop_queued_input() == "second"
        assert await store.pop_queued_input() is None


class TestInputReceiver:
    @pytest.fixture
    def store(self, kv_store):
        human_store = InputStore()
        with human_store.use_store(kv_store):
            yield human_store

    async def test_none_input_is_ignored(self, store):
        receiver = InputReceiver(store)

        await receiver.receive_input(None)

        assert await store.pop_queued_input() is None

    async def test_blank_input_is_ignored(self, store):
        receiver = InputReceiver(store)

        await receiver.receive_input("   ")

        assert await store.pop_queued_input() is None

    async def test_list_input_is_joined(self, store):
        receiver = InputReceiver(store)

        await receiver.receive_input(["hello", "world"])

        assert await store.pop_queued_input() == "hello world"

    async def test_input_is_queued_when_nothing_pending(self, store):
        receiver = InputReceiver(store)

        await receiver.receive_input("hello")

        assert await store.pop_queued_input() == "hello"

    async def test_single_pending_request_is_answered(self, store):
        await store.save_pending_requests({"only": _request("only")})
        receiver = InputReceiver(store)

        await receiver.receive_input("the answer")

        assert await store.get_input("only") == "the answer"

    async def test_multiple_pending_requires_reply_to(self, store):
        await store.save_pending_requests({"a": _request("a"), "b": _request("b")})
        receiver = InputReceiver(store)

        with pytest.raises(AmbiguousInputError) as exc_info:
            await receiver.receive_input("ambiguous")

        assert sorted(exc_info.value.interaction_ids) == ["a", "b"]

    async def test_reply_to_targets_specific_request(self, store):
        await store.save_pending_requests({"a": _request("a"), "b": _request("b")})
        receiver = InputReceiver(store)

        await receiver.receive_input("for b", reply_to="b")

        assert await store.get_input("b") == "for b"
        assert await store.get_input("a") is None

    async def test_reply_to_unknown_request_raises(self, store):
        await store.save_pending_requests({"a": _request("a")})
        receiver = InputReceiver(store)

        with pytest.raises(UnknownInputError) as exc_info:
            await receiver.receive_input("oops", reply_to="missing")

        assert exc_info.value.interaction_id == "missing"


class TestInputCoordinator:
    @pytest.fixture
    def coordinator(self, kv_store):
        coordinator = InputCoordinator()
        with coordinator.store.use_store(kv_store):
            yield coordinator

    async def test_record_request_records_pending_and_notifies(self, kv_store):
        seen: list[InputRequest] = []
        coordinator = InputCoordinator(on_request=seen.append)

        with coordinator.store.use_store(kv_store):
            await coordinator.record_request("x", "why?")

            pending = await coordinator.store.pending_requests()

        assert "x" in pending
        assert seen == [InputRequest(interaction_id="x", prompt="why?")]

    async def test_prompt_delta_notifies(self, kv_store):
        seen: list[str] = []
        coordinator = InputCoordinator(on_prompt_delta=seen.append)

        with coordinator.store.use_store(kv_store):
            await coordinator.notify_prompt_delta("What ")
            await coordinator.notify_prompt_delta("topic?")

        assert seen == ["What ", "topic?"]

    async def test_complete_request_removes_pending(self, coordinator):
        await coordinator.store.save_pending_requests({"x": _request("x")})

        await coordinator.complete_request("x")

        assert await coordinator.store.pending_requests() == {}

    async def test_provide_input_returns_stored_input(self, coordinator):
        await coordinator.store.set_input("x", "stored")

        assert await coordinator.provide_input("x") == "stored"

    async def test_provide_input_consumes_queued_input_when_unblocked(
        self, coordinator
    ):
        await coordinator.store.queue_input("queued")

        assert await coordinator.provide_input("x") == "queued"

    async def test_provide_input_does_not_consume_queue_with_other_pending(
        self, coordinator
    ):
        await coordinator.store.save_pending_requests({"other": _request("other")})
        await coordinator.store.queue_input("queued")

        assert await coordinator.provide_input("x") is None
        # The queued input is left untouched for later.
        assert await coordinator.store.pop_queued_input() == "queued"

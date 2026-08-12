import pytest
import sefia_typer
from sefia_typer import InputChannel, InputRequest
from sefia_typer._input import _to_input_text
from sefia_typer.exceptions import AmbiguousInputError, UnknownInputError


def test_exceptions_are_not_reexported_from_package_root():
    assert "AmbiguousInputError" not in sefia_typer.__all__
    assert "UnknownInputError" not in sefia_typer.__all__
    assert "UnknownSessionError" not in sefia_typer.__all__
    assert not hasattr(sefia_typer, "AmbiguousInputError")
    assert not hasattr(sefia_typer, "UnknownInputError")
    assert not hasattr(sefia_typer, "UnknownSessionError")


class TestToInputText:
    def test_plain_string_is_stripped(self):
        assert _to_input_text("  hello  ") == "hello"

    def test_list_is_joined_with_spaces(self):
        assert _to_input_text(["hello", "world"]) == "hello world"

    def test_list_result_is_stripped(self):
        assert _to_input_text(["  hello  "]) == "hello"

    def test_empty_list_is_empty_string(self):
        assert _to_input_text([]) == ""


@pytest.fixture
def channel(kv_store):
    channel = InputChannel()
    with channel.use_store(kv_store):
        yield channel


class TestBinding:
    async def test_requires_bound_store(self):
        channel = InputChannel()

        with pytest.raises(RuntimeError):
            await channel.pending()

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError, match="namespace"):
            InputChannel(namespace="/")

    async def test_namespace_scopes_persisted_keys(self, kv_store):
        channel = InputChannel(namespace="custom/input")

        with channel.use_store(kv_store):
            await channel.record_request("x", "why?")
            await channel.receive_input("answer", reply_to="x")

        assert "custom/input/pending" in kv_store._data
        assert kv_store._data["custom/input/input/x"] == "answer"
        assert "input_channel/pending" not in kv_store._data


class TestPending:
    async def test_empty_by_default(self, channel):
        assert await channel.pending() == []

    async def test_recorded_request_is_pending(self, channel):
        await channel.record_request("a", "prompt a?")

        assert await channel.pending() == [
            InputRequest(interaction_id="a", prompt="prompt a?")
        ]

    async def test_pending_is_ordered_by_interaction_id(self, channel):
        await channel.record_request("b", "prompt b?")
        await channel.record_request("a", "prompt a?")

        pending = await channel.pending()

        assert [request.interaction_id for request in pending] == ["a", "b"]

    async def test_resolved_requests_are_dropped_from_pending(self, channel):
        await channel.record_request("a", "prompt a?")
        await channel.record_request("b", "prompt b?")

        await channel.receive_input("resolved", reply_to="a")

        pending = await channel.pending()
        assert [request.interaction_id for request in pending] == ["b"]

    async def test_complete_request_removes_pending(self, channel):
        await channel.record_request("x", "why?")

        await channel.complete_request("x")

        assert await channel.pending() == []

    async def test_record_request_notifies(self, kv_store):
        seen: list[InputRequest] = []
        channel = InputChannel(on_request=seen.append)

        with channel.use_store(kv_store):
            await channel.record_request("x", "why?")

        assert seen == [InputRequest(interaction_id="x", prompt="why?")]

    async def test_prompt_delta_notifies(self, kv_store):
        seen: list[tuple[str, str]] = []
        channel = InputChannel(
            on_prompt_delta=lambda call_id, text: seen.append((call_id, text))
        )

        with channel.use_store(kv_store):
            await channel.notify_prompt_delta("call-1", "What ")
            await channel.notify_prompt_delta("call-1", "topic?")

        assert seen == [("call-1", "What "), ("call-1", "topic?")]


class TestReceiveInput:
    async def test_none_input_is_ignored(self, channel):
        await channel.receive_input(None)

        assert await channel.provide_input("any") is None

    async def test_blank_input_is_ignored(self, channel):
        await channel.receive_input("   ")

        assert await channel.provide_input("any") is None

    async def test_list_input_is_joined(self, channel):
        await channel.receive_input(["hello", "world"])

        assert await channel.provide_input("any") == "hello world"

    async def test_input_is_queued_when_nothing_pending(self, channel):
        await channel.receive_input("hello")

        assert await channel.provide_input("any") == "hello"

    async def test_single_pending_request_is_resolved(self, channel):
        await channel.record_request("only", "prompt?")

        await channel.receive_input("the input")

        assert await channel.provide_input("only") == "the input"

    async def test_multiple_pending_requires_reply_to(self, channel):
        await channel.record_request("a", "prompt a?")
        await channel.record_request("b", "prompt b?")

        with pytest.raises(AmbiguousInputError) as exc_info:
            await channel.receive_input("ambiguous")

        assert sorted(exc_info.value.interaction_ids) == ["a", "b"]

    async def test_reply_to_targets_specific_request(self, channel):
        await channel.record_request("a", "prompt a?")
        await channel.record_request("b", "prompt b?")

        await channel.receive_input("for b", reply_to="b")

        assert await channel.provide_input("b") == "for b"

    async def test_reply_to_unknown_request_raises(self, channel):
        await channel.record_request("a", "prompt a?")

        with pytest.raises(UnknownInputError) as exc_info:
            await channel.receive_input("oops", reply_to="missing")

        assert exc_info.value.interaction_id == "missing"


class TestProvideInput:
    async def test_returns_none_when_nothing_available(self, channel):
        assert await channel.provide_input("x") is None

    async def test_queued_inputs_are_claimed_in_order(self, channel):
        await channel.receive_input("first")
        await channel.receive_input("second")

        assert await channel.provide_input("i1") == "first"
        assert await channel.provide_input("i2") == "second"
        assert await channel.provide_input("i3") is None

    async def test_does_not_claim_queue_with_other_pending(self, channel):
        await channel.receive_input("queued")
        await channel.record_request("other", "other prompt?")

        assert await channel.provide_input("mine") is None
        # The queued input is left for the pending request.
        assert await channel.provide_input("other") == "queued"

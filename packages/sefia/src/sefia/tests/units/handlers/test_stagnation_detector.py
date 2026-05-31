from datetime import datetime

import pytest

from sefia.events import BeforeToolCall
from sefia.handlers.stagnation import StagnationDetector, StagnationError
from sefia.models import ToolCallRequest


class TestStagnationDetector:
    def create_event(self, name: str, args: dict) -> BeforeToolCall:
        return BeforeToolCall(
            tool_call=ToolCallRequest(id="1", name=name, arguments=args)
        )

    async def test_raises_error_on_repeated_calls(self):
        detector = StagnationDetector(max_repeats=3)
        event = self.create_event("test_tool", {"a": 1})

        await detector.handle(event)
        await detector.handle(event)

        with pytest.raises(StagnationError):
            await detector.handle(event)

    async def test_does_not_raise_error_for_different_calls(self):
        detector = StagnationDetector(max_repeats=2)
        event1 = self.create_event("test_tool", {"a": 1})
        event2 = self.create_event("test_tool", {"a": 2})

        await detector.handle(event1)
        await detector.handle(event2)  # Should not raise

    async def test_history_resets_after_different_call(self):
        detector = StagnationDetector(max_repeats=2)
        event1 = self.create_event("tool1", {"a": 1})
        event2 = self.create_event("tool2", {"b": 2})

        await detector.handle(event1)
        await detector.handle(event2)
        await detector.handle(event1)  # Should not raise, history was broken by event2

    async def test_does_not_raise_if_limit_not_reached(self):
        detector = StagnationDetector(max_repeats=3)
        event = self.create_event("test_tool", {"a": 1})

        await detector.handle(event)
        await detector.handle(event)  # Should not raise

    def test_hashes_nested_dictionaries_consistently(self):
        detector = StagnationDetector()
        args1 = {"a": 1, "nested": {"c": 3, "b": 2}}
        args2 = {"nested": {"b": 2, "c": 3}, "a": 1}

        hash1 = detector._hash_tool_call("test_tool", args1)
        hash2 = detector._hash_tool_call("test_tool", args2)

        assert hash1 == hash2

    def test_handles_non_serializable_types_with_fallback(self):
        detector = StagnationDetector()
        now = datetime.now()
        args = {"dt": now, "num": 1}

        try:
            # Should not raise TypeError
            h = detector._hash_tool_call("test_tool", args)
            assert str(now) in h
        except Exception as e:
            pytest.fail(f"Hashing with non-serializable type failed: {e}")

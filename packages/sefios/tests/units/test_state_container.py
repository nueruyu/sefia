from dataclasses import dataclass

import pytest
from pytest_mock import MockerFixture

from sefios import StateContainer, StateRegistry, get_state, state
from sefios.state import _default_registry


@dataclass
class _SampleState:
    value: int = 0


class TestStateRegistry:
    def test_register_and_lookup(self):
        registry = StateRegistry()
        registry.register(_SampleState, "sample.key")
        assert registry.key_for(_SampleState) == "sample.key"

    def test_re_registering_same_type_raises(self):
        registry = StateRegistry()
        registry.register(_SampleState, "sample.key")
        # A type may only be registered once, even under the same key.
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_SampleState, "sample.key")

    def test_registering_same_type_under_different_key_raises(self):
        registry = StateRegistry()
        registry.register(_SampleState, "sample.key")
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_SampleState, "other.key")

    def test_allows_same_key_for_reimported_type(self):
        # Simulate a double import: two distinct class objects with the same
        # __qualname__ (as if one class were imported under two module paths).
        def _make():
            @dataclass
            class Reimported:
                value: int = 0

            return Reimported

        type_a = _make()
        type_b = _make()
        assert type_a is not type_b

        registry = StateRegistry()
        registry.register(type_a, "shared.key")
        registry.register(type_b, "shared.key")
        assert registry.key_for(type_a) == "shared.key"
        assert registry.key_for(type_b) == "shared.key"

    def test_registering_key_used_by_different_type_raises(self):
        @dataclass
        class _OtherState:
            value: int = 0

        registry = StateRegistry()
        registry.register(_SampleState, "shared.key")
        with pytest.raises(ValueError, match="different type"):
            registry.register(_OtherState, "shared.key")

    def test_key_for_unregistered_type_raises(self):
        registry = StateRegistry()
        with pytest.raises(KeyError, match="not a .*registered state type"):
            registry.key_for(_SampleState)


class TestStateDecorator:
    def test_decorator_registers_in_default_registry(self):
        @state(key="decorated.key")
        @dataclass
        class _DecoratedState:
            value: int = 0

        assert _default_registry.key_for(_DecoratedState) == "decorated.key"


class TestStateContainer:
    def test_get_resolves_key_and_delegates_to_bound_state(self, mocker: MockerFixture):
        registry = StateRegistry()
        registry.register(_SampleState, "sample.key")
        sentinel_store = object()
        get_state_store = mocker.patch(
            "sefios.state.get_state_store", return_value=sentinel_store
        )

        container = StateContainer(registry=registry)
        result = container.get(_SampleState)

        assert result is sentinel_store
        get_state_store.assert_called_once_with("sample.key", _SampleState)


class TestGetState:
    def test_raises_outside_session(self):
        with pytest.raises(RuntimeError, match="No sefios session state is bound"):
            get_state()

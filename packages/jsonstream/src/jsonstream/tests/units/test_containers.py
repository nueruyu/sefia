from jsonstream._containers import ContainerTracker
from jsonstream.events import (
    EndArray,
    EndObject,
    JsonParseError,
    StartArray,
    StartObject,
)


def collect_events(tracker: ContainerTracker, char: str):
    return list(tracker.parse_structural_char(char))


def test_tracks_nested_object_array_paths_and_completion():
    tracker = ContainerTracker()

    assert collect_events(tracker, "{") == [StartObject(path=())]
    assert tracker.is_expecting_object_key()

    assert tracker.set_object_key("items")
    assert not tracker.is_expecting_object_key()
    assert not tracker.is_expecting_value()

    assert collect_events(tracker, ":") == []
    assert tracker.is_expecting_value()

    assert collect_events(tracker, "[") == [StartArray(path=("items",))]
    assert tracker.path == ("items", 0)

    tracker.value_completed()
    assert tracker.path == ("items", 1)

    assert collect_events(tracker, "]") == [EndArray(path=("items", 1))]
    assert tracker.path == ("items",)

    assert collect_events(tracker, "}") == [EndObject(path=("items",))]
    assert not tracker.has_unclosed_containers


def test_comma_moves_object_back_to_key_state():
    tracker = ContainerTracker()

    assert collect_events(tracker, "{") == [StartObject(path=())]
    assert tracker.set_object_key("a")
    tracker.value_completed()
    assert not tracker.is_expecting_object_key()

    assert collect_events(tracker, ",") == []
    assert tracker.is_expecting_object_key()


def test_comma_moves_array_back_to_value_state():
    tracker = ContainerTracker()

    assert collect_events(tracker, "[") == [StartArray(path=())]
    tracker.value_completed()
    assert not tracker.is_expecting_value()

    assert collect_events(tracker, ",") == []
    assert tracker.is_expecting_value()


def test_rejects_root_array_end_after_root_value_completed():
    tracker = ContainerTracker()
    tracker.value_completed()

    events = collect_events(tracker, "]")

    assert events == [JsonParseError("Unexpected ']'", fatal=True)]

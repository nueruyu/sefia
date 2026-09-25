import pytest
from sefia.llm.json import JsonCompatible, JsonSnapshot


def test_capture_builds_nested_snapshot() -> None:
    value = JsonSnapshot.capture(
        {"name": "report", "items": [1, True, None], "metadata": {"count": 3}}
    )

    fields = value.as_object()
    assert fields["name"].as_string() == "report"
    assert [item.as_scalar() for item in fields["items"].as_array()] == [
        1,
        True,
        None,
    ]
    assert fields["metadata"].as_object()["count"].as_scalar() == 3
    assert value.to_json_compatible() == {
        "name": "report",
        "items": [1, True, None],
        "metadata": {"count": 3},
    }


def test_parse_json_builds_validated_snapshot() -> None:
    value = JsonSnapshot.parse_json('{"items": [1, true, null]}')

    assert value.to_json_compatible() == {"items": [1, True, None]}


def test_parse_json_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        JsonSnapshot.parse_json("not json")


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("as_object", "must be an object"),
        ("as_array", "must be an array"),
        ("as_string", "must be a string"),
    ],
)
def test_shape_accessors_reject_wrong_shape(method: str, message: str) -> None:
    value = JsonSnapshot.from_scalar(1)

    with pytest.raises(ValueError, match=message):
        getattr(value, method)()


def test_to_scalar_rejects_container() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        JsonSnapshot.from_array([]).as_scalar()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (JsonSnapshot.from_scalar("value"), "value"),
        (JsonSnapshot.from_array([JsonSnapshot.from_scalar(1)]), [1]),
        (
            JsonSnapshot.from_object({"field": JsonSnapshot.from_scalar(True)}),
            {"field": True},
        ),
    ],
)
def test_to_json_compatible_projects_each_tree_shape(
    value: JsonSnapshot, expected: object
) -> None:
    assert value.to_json_compatible() == expected


def test_capture_owns_nested_input() -> None:
    source: JsonCompatible = {"items": [{"value": 1}]}
    data = JsonSnapshot.capture(source)

    assert isinstance(source, dict)
    items = source["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["value"] = 2

    assert data.to_json_compatible() == {"items": [{"value": 1}]}


def test_projection_returns_detached_nested_containers() -> None:
    data = JsonSnapshot.capture({"items": [{"value": 1}]})

    tree = data.to_json_compatible()
    assert isinstance(tree, dict)
    items = tree["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["value"] = 2
    items.append(None)

    assert data.to_json_compatible() == {"items": [{"value": 1}]}


def test_from_array_does_not_retain_mutable_aliases() -> None:
    child = JsonSnapshot.capture({"items": [1]})
    data = JsonSnapshot.from_array([child])
    child_tree = child.to_json_compatible()
    assert isinstance(child_tree, dict)
    items = child_tree["items"]
    assert isinstance(items, list)
    items.append(2)

    assert data.to_json_compatible() == [{"items": [1]}]


def test_from_object_does_not_retain_mapping_or_value_aliases() -> None:
    child = JsonSnapshot.capture({"items": [1]})
    fields = {"child": child}
    data = JsonSnapshot.from_object(fields)
    fields["child"] = JsonSnapshot.from_scalar("changed")
    child_tree = child.to_json_compatible()
    assert isinstance(child_tree, dict)
    items = child_tree["items"]
    assert isinstance(items, list)
    items.append(2)

    assert data.to_json_compatible() == {"child": {"items": [1]}}


def test_value_equality_and_unhashability() -> None:
    first = JsonSnapshot.capture({"items": [1]})
    assert first == JsonSnapshot.parse_json('{"items": [1]}')
    assert first != JsonSnapshot.capture({"items": [2]})
    with pytest.raises(TypeError):
        hash(first)


@pytest.mark.parametrize(
    "value", [{1: "one"}, {"nested": {None: 2}}, {"bad": object()}]
)
def test_capture_rejects_invalid_json(value: object) -> None:
    from typing import cast

    with pytest.raises(ValueError):
        JsonSnapshot.capture(cast(JsonCompatible, value))


class _AliasingList(list[JsonCompatible]):
    def __deepcopy__(self, memo: dict[int, object]) -> "_AliasingList":
        return self


class _AliasingDict(dict[str, JsonCompatible]):
    def __deepcopy__(self, memo: dict[int, object]) -> "_AliasingDict":
        return self


def test_capture_detaches_container_subclasses_without_copy_hooks() -> None:
    items = _AliasingList([1])
    nested = _AliasingDict({"items": items})
    source = _AliasingList([nested])
    snapshot = JsonSnapshot.capture(source)
    items.append(2)
    nested["added"] = True
    source.append(None)

    projected = snapshot.to_json_compatible()
    assert projected == [{"items": [1]}]
    assert type(projected) is list
    assert type(projected[0]) is dict
    assert type(projected[0]["items"]) is list
    projected[0]["items"].append(3)
    assert snapshot.to_json_compatible() == [{"items": [1]}]


def test_projection_detaches_owned_container_subclasses() -> None:
    snapshot = JsonSnapshot._from_owned(_AliasingDict({"items": _AliasingList([1])}))
    projected = snapshot.to_json_compatible()
    assert isinstance(projected, dict)
    items = projected["items"]
    assert isinstance(items, list)
    items.append(2)
    assert snapshot.to_json_compatible() == {"items": [1]}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_snapshot_constructors_reject_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        JsonSnapshot.from_scalar(value)
    with pytest.raises(ValueError, match="must be finite"):
        JsonSnapshot.capture({"nested": [value]})
    with pytest.raises(ValueError, match="must be finite"):
        JsonSnapshot._from_owned({"nested": [value]})


@pytest.mark.parametrize(
    "text", ["NaN", "Infinity", "-Infinity", "1e999", '{"nested": [NaN]}']
)
def test_parse_json_rejects_non_finite_numbers(text: str) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        JsonSnapshot.parse_json(text)

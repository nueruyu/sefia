import pytest

from sefia.llm.structured_data import StructuredData, StructuredDataTree
from sefia.llm.json_schema import JsonScalar, JsonValue


def test_from_json_builds_nested_structured_data() -> None:
    value = StructuredData.from_json(
        {"name": "report", "items": [1, True, None], "metadata": {"count": 3}}
    )

    fields = value.to_object()
    assert fields["name"].to_string() == "report"
    assert [item.to_scalar() for item in fields["items"].to_array()] == [
        1,
        True,
        None,
    ]
    assert fields["metadata"].to_object()["count"].to_scalar() == 3
    assert value.tree == {
        "name": "report",
        "items": [1, True, None],
        "metadata": {"count": 3},
    }


def test_parse_json_builds_validated_structured_data() -> None:
    value = StructuredData.parse_json('{"items": [1, true, null]}')

    assert value.tree == {"items": [1, True, None]}


def test_parse_json_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        StructuredData.parse_json("not json")


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("to_object", "must be an object"),
        ("to_array", "must be an array"),
        ("to_string", "must be a string"),
    ],
)
def test_shape_accessors_reject_wrong_shape(method: str, message: str) -> None:
    value = StructuredData.from_scalar(1)

    with pytest.raises(ValueError, match=message):
        getattr(value, method)()


def test_to_object_rejects_mapping_keys() -> None:
    value = StructuredData.from_mapping({1: StructuredData.from_scalar("one")})

    with pytest.raises(ValueError, match="must have string keys"):
        value.to_object("record")


def test_to_scalar_rejects_container() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        StructuredData.from_array([]).to_scalar()


def test_to_json_value_projects_nested_scalar_keys() -> None:
    value = StructuredData.from_mapping(
        {
            None: StructuredData.from_scalar("none"),
            True: StructuredData.from_array([StructuredData.from_scalar(1)]),
            2.5: StructuredData.from_mapping(
                {False: StructuredData.from_scalar("false")}
            ),
        }
    )

    assert value.to_json_value() == {
        "null": "none",
        "true": [1],
        "2.5": {"false": "false"},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (StructuredData.from_scalar("value"), "value"),
        (StructuredData.from_array([StructuredData.from_scalar(1)]), [1]),
        (
            StructuredData.from_object({"field": StructuredData.from_scalar(True)}),
            {"field": True},
        ),
    ],
)
def test_to_json_value_projects_each_tree_shape(
    value: StructuredData, expected: object
) -> None:
    assert value.to_json_value() == expected


@pytest.mark.parametrize(
    "entries",
    [
        {
            1: StructuredData.from_scalar("number"),
            "1": StructuredData.from_scalar("text"),
        },
        {
            None: StructuredData.from_scalar("none"),
            "null": StructuredData.from_scalar("text"),
        },
        {
            True: StructuredData.from_scalar("bool"),
            "true": StructuredData.from_scalar("text"),
        },
    ],
)
def test_to_json_value_rejects_key_collisions(
    entries: dict[JsonScalar, StructuredData],
) -> None:
    value = StructuredData.from_mapping(entries)

    with pytest.raises(ValueError, match="same JSON key"):
        value.to_json_value()


def test_from_tree_owns_nested_input() -> None:
    source: StructuredDataTree = {"items": [{"value": 1}]}
    data = StructuredData.from_tree(source)

    assert isinstance(source, dict)
    items = source["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["value"] = 2

    assert data.tree == {"items": [{"value": 1}]}


def test_from_json_owns_nested_input() -> None:
    source: JsonValue = {"items": [{"value": 1}]}
    data = StructuredData.from_json(source)

    assert isinstance(source, dict)
    items = source["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["value"] = 2

    assert data.tree == {"items": [{"value": 1}]}


def test_tree_returns_a_detached_nested_projection() -> None:
    data = StructuredData.from_json({"items": [{"value": 1}]})

    tree = data.tree
    assert isinstance(tree, dict)
    items = tree["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["value"] = 2
    items.append(None)

    assert data.tree == {"items": [{"value": 1}]}


def test_from_array_does_not_retain_mutable_aliases() -> None:
    child = StructuredData.from_json({"items": [1]})
    data = StructuredData.from_array([child])
    child_tree = child.tree
    assert isinstance(child_tree, dict)
    items = child_tree["items"]
    assert isinstance(items, list)
    items.append(2)

    assert data.tree == [{"items": [1]}]


def test_from_object_does_not_retain_mapping_or_value_aliases() -> None:
    child = StructuredData.from_json({"items": [1]})
    fields = {"child": child}
    data = StructuredData.from_object(fields)
    fields["child"] = StructuredData.from_scalar("changed")
    child_tree = child.tree
    assert isinstance(child_tree, dict)
    items = child_tree["items"]
    assert isinstance(items, list)
    items.append(2)

    assert data.tree == {"child": {"items": [1]}}


def test_from_mapping_does_not_retain_mapping_or_value_aliases() -> None:
    child = StructuredData.from_json({"items": [1]})
    entries: dict[JsonScalar, StructuredData] = {1: child}
    data = StructuredData.from_mapping(entries)
    entries[1] = StructuredData.from_scalar("changed")
    child_tree = child.tree
    assert isinstance(child_tree, dict)
    items = child_tree["items"]
    assert isinstance(items, list)
    items.append(2)

    assert data.tree == {1: {"items": [1]}}

import pytest

from sefia.llm.structured_value import StructuredValue


def test_from_json_builds_nested_structured_value() -> None:
    value = StructuredValue.from_json(
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
    assert value.value == {
        "name": "report",
        "items": [1, True, None],
        "metadata": {"count": 3},
    }


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("to_object", "must be an object"),
        ("to_array", "must be an array"),
        ("to_string", "must be a string"),
    ],
)
def test_shape_accessors_reject_wrong_shape(method: str, message: str) -> None:
    value = StructuredValue.from_scalar(1)

    with pytest.raises(ValueError, match=message):
        getattr(value, method)()


def test_to_object_rejects_mapping_keys() -> None:
    value = StructuredValue.from_mapping({1: StructuredValue.from_scalar("one")})

    with pytest.raises(ValueError, match="must have string keys"):
        value.to_object("record")


def test_to_scalar_rejects_container() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        StructuredValue.from_array([]).to_scalar()

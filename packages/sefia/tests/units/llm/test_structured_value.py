import pytest

from sefia.llm.structured_value import StructuredValue


def test_from_json_builds_nested_structured_value() -> None:
    value = StructuredValue.from_json(
        {"name": "report", "items": [1, True, None], "metadata": {"count": 3}}
    )

    fields = value.as_record()
    assert fields["name"].as_string() == "report"
    assert [item.as_scalar() for item in fields["items"].as_array()] == [
        1,
        True,
        None,
    ]
    assert fields["metadata"].as_record()["count"].as_scalar() == 3
    assert value.to_python() == {
        "name": "report",
        "items": [1, True, None],
        "metadata": {"count": 3},
    }


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("as_object", "must be an object"),
        ("as_array", "must be an array"),
        ("as_string", "must be a string"),
    ],
)
def test_shape_accessors_reject_wrong_shape(method: str, message: str) -> None:
    value = StructuredValue.scalar(1)

    with pytest.raises(ValueError, match=message):
        getattr(value, method)()


def test_as_record_rejects_non_string_keys() -> None:
    value = StructuredValue.object({1: StructuredValue.scalar("one")})

    with pytest.raises(ValueError, match="must have string keys"):
        value.as_record("record")


def test_as_scalar_rejects_container() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        StructuredValue.array([]).as_scalar()

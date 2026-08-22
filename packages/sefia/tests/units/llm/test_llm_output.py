import pytest

from sefia.llm.llm_output import LLMOutput


def test_from_json_builds_nested_llm_output() -> None:
    value = LLMOutput.from_json(
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
    assert value.data == {
        "name": "report",
        "items": [1, True, None],
        "metadata": {"count": 3},
    }


def test_parse_json_builds_validated_llm_output() -> None:
    value = LLMOutput.parse_json('{"items": [1, true, null]}')

    assert value.data == {"items": [1, True, None]}


def test_parse_json_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        LLMOutput.parse_json("not json")


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("to_object", "must be an object"),
        ("to_array", "must be an array"),
        ("to_string", "must be a string"),
    ],
)
def test_shape_accessors_reject_wrong_shape(method: str, message: str) -> None:
    value = LLMOutput.from_scalar(1)

    with pytest.raises(ValueError, match=message):
        getattr(value, method)()


def test_to_object_rejects_mapping_keys() -> None:
    value = LLMOutput.from_mapping({1: LLMOutput.from_scalar("one")})

    with pytest.raises(ValueError, match="must have string keys"):
        value.to_object("record")


def test_to_scalar_rejects_container() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        LLMOutput.from_array([]).to_scalar()

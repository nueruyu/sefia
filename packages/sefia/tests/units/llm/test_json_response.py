import json

import pytest

from sefia.llm._json_response import parse_json_response


def test_parses_strict_json():
    output = parse_json_response('{"value":1}', allow_surrounding_text=False)

    assert output.data == {"value": 1}


def test_parses_a_leading_code_fence_for_existing_clients():
    output = parse_json_response(
        '```json\n{"value":1}\n```', allow_surrounding_text=False
    )

    assert output.data == {"value": 1}


def test_extracts_fenced_json_surrounded_by_prose():
    output = parse_json_response(
        'I will call it.\n```json\n{"decision":"tool_calls"}\n```\nDone.',
        allow_surrounding_text=True,
    )

    assert output.data == {"decision": "tool_calls"}


def test_extracts_json_wrapped_in_xml():
    output = parse_json_response(
        '<function_calls>\n{"decision":"tool_calls"}\n</function_calls>',
        allow_surrounding_text=True,
    )

    assert output.data == {"decision": "tool_calls"}


def test_uses_only_the_first_complete_json_value():
    output = parse_json_response(
        'first {"decision":"tool_calls"} then {"decision":"result"}',
        allow_surrounding_text=True,
    )

    assert output.data == {"decision": "tool_calls"}


def test_does_not_extract_scalar_values_from_prose():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("There are 2. Result: 42", allow_surrounding_text=True)


def test_does_not_extract_surrounded_json_without_opt_in():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response('prose {"decision":"result"}', allow_surrounding_text=False)

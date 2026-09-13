"""GLM argument loss and nullable-string coercion regressions."""

import pytest
from transformers import AutoTokenizer

from renderers.base import ToolCallParseStatus
from renderers.parsing import _coerce_arg_value, parse_glm


@pytest.mark.parametrize(
    "schema",
    [
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"anyOf": [{"type": "null"}, {"type": "string"}]},
        {"oneOf": [{"type": "string"}, {"type": "null"}]},
        {"type": ["string", "null"]},
        {"type": ["null", "string"]},
    ],
)
@pytest.mark.parametrize(
    "text", ["1e3", "42", "true", "false", "[1]", '{"x":1}', " 123 \n"]
)
def test_nullable_string_does_not_coerce_other_json_types(schema, text):
    assert _coerce_arg_value(text, schema) == (text, False)
    assert _coerce_arg_value(" null \n", schema) == (None, False)


@pytest.mark.parametrize(
    ("schema", "text", "expected"),
    [
        ({"type": "string"}, "null", "null"),
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "42", 42),
        ({"oneOf": [{"type": "string"}, {"type": "boolean"}]}, "true", True),
        ({"anyOf": [{"type": ["string", "integer"]}, {"type": "null"}]}, "42", 42),
        (None, "1e3", 1000.0),
    ],
)
def test_other_argument_coercion_is_unchanged(schema, text, expected):
    actual, fallback = _coerce_arg_value(text, schema)
    assert actual == expected
    assert type(actual) is type(expected)
    assert not fallback


@pytest.fixture(scope="module")
def glm_tokenizer():
    return AutoTokenizer.from_pretrained(
        "zai-org/GLM-5.3-Flash",
        revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    )


@pytest.mark.parametrize(
    "broken",
    [
        "<arg_key>command</arg_key>printf bad</arg_value>",
        "<arg_key>command<arg_value>printf bad</arg_value>",
    ],
)
def test_glm_does_not_borrow_delimiters_from_the_next_argument(glm_tokenizer, broken):
    parsed = _parse(
        glm_tokenizer,
        "<arg_key>workdir</arg_key><arg_value>/workspace</arg_value>"
        + broken
        + "<arg_key>timeout</arg_key><arg_value>1000</arg_value>",
    )
    assert parsed.status == ToolCallParseStatus.MALFORMED_STRUCTURE
    assert parsed.arguments == {"workdir": "/workspace"}


@pytest.mark.parametrize(
    "command", ["printf ok", "printf '<arg_key>literal</arg_key>'", "  \n"]
)
def test_glm_valid_value_text_and_adjacent_arguments_are_preserved(
    glm_tokenizer, command
):
    parsed = _parse(
        glm_tokenizer,
        "<arg_key>command</arg_key><arg_value>"
        + command
        + "</arg_value><arg_key>timeout</arg_key><arg_value>1000</arg_value>",
    )
    assert parsed.status == ToolCallParseStatus.OK
    assert parsed.arguments == {"command": command, "timeout": 1000}


def _parse(tokenizer, arguments):
    ids = tokenizer.encode(
        "Inspect.</think><tool_call>bash" + arguments + "</tool_call>",
        add_special_tokens=False,
    )
    token_id = tokenizer.convert_tokens_to_ids
    result = parse_glm(
        tokenizer,
        ids,
        stop_ids=set(),
        think_id=token_id("<think>"),
        think_end_id=token_id("</think>"),
        tool_call_id=token_id("<tool_call>"),
        tool_call_end_id=token_id("</tool_call>"),
        arg_key_id=token_id("<arg_key>"),
        arg_key_end_id=token_id("</arg_key>"),
        arg_value_id=token_id("<arg_value>"),
        arg_value_end_id=token_id("</arg_value>"),
        tools=[
            {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "workdir": {"type": "string"},
                    },
                },
            }
        ],
    )
    return result.tool_calls[0]

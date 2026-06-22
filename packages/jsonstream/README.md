# jsonstream

Incremental JSON parsing for code that receives JSON in chunks.

`jsonstream` consumes text incrementally and emits parse events as soon as they
are available. It is useful for streaming model responses, network payloads, or
other producers where you want to react before the full JSON document has been
received.

## What it does

- Parses JSON from one or more string chunks.
- Emits structural events for objects and arrays.
- Emits scalar events for numbers, booleans, and `null`.
- Emits string values incrementally with `StringDelta` events.
- Tracks a path for every event using object keys and array indexes.
- Reports parse failures as `JsonParseError` events instead of raising for normal
  parse errors.

## Installation

When using the workspace:

```bash
uv sync --all-packages
```

For package use after publishing:

```bash
pip install jsonstream
```

## Basic usage

Import the parser from the package root. Import events from `jsonstream.events`.

```python
from jsonstream import IncrementalJsonParser
from jsonstream.events import EndString, JsonParseError, StringDelta

parser = IncrementalJsonParser()

chunks = [
    '{"message": "hel',
    'lo", "count": 2}',
]

for chunk in chunks:
    for event in parser.feed(chunk):
        if isinstance(event, StringDelta):
            print("string chunk:", event.path, event.text)
        elif isinstance(event, EndString):
            print("string value:", event.path, event.value)
        elif isinstance(event, JsonParseError):
            raise ValueError(event.message)

for event in parser.finish():
    if isinstance(event, JsonParseError):
        raise ValueError(event.message)
```

## Events

Every event has a `path` except `JsonParseError`. Paths are tuples containing
object keys and array indexes.

For this JSON:

```json
{"items": [{"name": "book"}]}
```

the string `"book"` is reported at:

```python
("items", 0, "name")
```

The event classes are:

- `StartObject`
- `EndObject`
- `StartArray`
- `EndArray`
- `Key`
- `StartString`
- `StringDelta`
- `EndString`
- `Scalar`
- `JsonParseError`

## String streaming

Strings are split into `StartString`, zero or more `StringDelta` events, and
`EndString`. Deltas are emitted at chunk boundaries and before escape sequences
are decoded.

```python
from jsonstream import IncrementalJsonParser
from jsonstream.events import StringDelta

parser = IncrementalJsonParser()

events = []
events.extend(parser.feed('["hel'))
events.extend(parser.feed('lo"]'))
events.extend(parser.finish())

assert StringDelta(path=(0,), text="hel") in events
assert StringDelta(path=(0,), text="lo") in events
```

Object keys are reported with `Key` events. They do not emit string delta events.

## Errors

Invalid input produces `JsonParseError(message, fatal=True)` events. After a
fatal error, callers should stop feeding input and discard the parser instance.

```python
from jsonstream import IncrementalJsonParser
from jsonstream.events import JsonParseError

parser = IncrementalJsonParser()
events = list(parser.feed("[1,]")) + list(parser.finish())

errors = [event for event in events if isinstance(event, JsonParseError)]
assert errors
```

## Development

Run the package tests:

```bash
uv run pytest packages/jsonstream
```

Run the full workspace tests and type check:

```bash
uv run pytest
uv run pyright
```

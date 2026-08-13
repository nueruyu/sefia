# 03 FastAPI API

A REST API built with [FastAPI](https://fastapi.tiangolo.com/) that serves a
human-in-the-loop `@infer` workflow. Lifecycle updates are streamed through a
session event channel, while the workflow itself stays a normal HTTP
request/response endpoint.

## Run

Run from the repository root. See the [examples README](../README.md) for setup
(`OPENAI_API_KEY`, optional `EXAMPLE_DEFAULT_MODEL`).

```bash
uv run python -m examples.03_fastapi_api
```

The server listens on `http://127.0.0.1:8000`. Open `/` for a small HITL chat UI,
or `/docs` for the interactive API docs.

## Browser UI

Open `http://127.0.0.1:8000/` for a dependency-free chat UI that creates a
session and sends messages to the human-in-the-loop interview endpoint. As the
agent composes a question its prompt is typed out live from `delta` events;
when the agent needs more detail, the `input_required` event finalizes that
prompt and the UI remembers its `interaction_id` for the next user reply. When
the workflow completes, the UI renders the structured brief.

The demo interviewer asks at most one focused follow-up question. If the request
is usable with reasonable assumptions, it completes the brief instead of asking
separately for topic, goal, and audience.

The UI streams the `delta` event — the *parsed* prompt and message text — rather
than the raw `@infer` response, so the internal structured envelope never
reaches the browser. Each delta carries `type` (`input` for a `get_input`
prompt, `output` for a `send_output` message) and the `interaction_id` of the
bubble it belongs to; the discrete `input_required` / `output` events close that
bubble.

The UI is plain HTML served by the FastAPI example, so it remains easy to inspect
and does not require a separate frontend toolchain.

## Endpoints

| Method & path | Purpose |
| --- | --- |
| `GET /` | Open the dependency-free HITL chat UI |
| `POST /sessions` | Create a persisted session |
| `GET /sessions/{id}/events` | Subscribe to session events via SSE |
| `POST /sessions/{id}/interview` | Run the human-in-the-loop interviewer |

## Human-in-the-loop (pause / resume)

The interview agent may pause to ask one clarifying question. The first POST
sends the initial request; if the agent needs more, the normal HTTP response is
`input_required` with an `interaction_id`. The same event is also published on
the session event stream when a client is subscribed.

```bash
SID=$(curl -s -X POST localhost:8000/sessions | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

curl -s -X POST localhost:8000/sessions/$SID/interview \
  -H 'content-type: application/json' \
  -d '{"input": "I want an article about our new product."}'

curl -s -X POST localhost:8000/sessions/$SID/interview \
  -H 'content-type: application/json' \
  -d '{"input": "Developers evaluating the product.", "reply_to": "<interaction_id>"}'
```

Browser clients can use the standard `EventSource` API for lifecycle events:

```js
const events = new EventSource(`/sessions/${sessionId}/events`)
events.addEventListener("delta", (event) => console.log(JSON.parse(event.data)))
events.addEventListener("completed", (event) => console.log(JSON.parse(event.data)))
events.addEventListener("input_required", (event) => console.log(JSON.parse(event.data)))
events.addEventListener("error", (event) => console.error(JSON.parse(event.data)))
```

## What it shows

- Writing FastAPI endpoints as ordinary HTTP request handlers
- Running a Sefia workflow with a CLI-like procedural session block
- Streaming parsed `delta` text (with a `type` of `input`/`output`) and publishing `completed`, `input_required`, and `error` events through a separate SSE channel
- Keeping SSE wiring out of the application workflow body
- Serving a dependency-free HITL chat UI from the same FastAPI process

## Note on sessions

Each session persists to disk under the example's `.local/` directory. Send
requests for a given session one at a time; concurrent turns against the *same*
session are not supported. Different sessions are independent and safe to run
concurrently: the shared input store binds the active session per task
(via a `ContextVar`), so overlapping requests do not see each other's state.

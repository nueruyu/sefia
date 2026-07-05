# 03 FastAPI API

A REST API built with [FastAPI](https://fastapi.tiangolo.com/) that serves the
same kind of `@infer` agents the CLI examples run. LLM tokens and lifecycle
updates are streamed through a session event channel, not by changing each
workflow endpoint into a streaming endpoint.

## Run

Run from the repository root. See the [examples README](../README.md) for setup
(`OPENAI_API_KEY`, optional `EXAMPLE_DEFAULT_MODEL`).

```bash
python -m examples.03_fastapi_api
```

The server listens on `http://127.0.0.1:8000` (interactive docs at `/docs`).

## Endpoints

| Method & path | Purpose |
| --- | --- |
| `POST /sessions` | Create a persisted session |
| `GET /sessions/{id}/events` | Subscribe to session events via SSE |
| `POST /sessions/{id}/answer` | Ask the one-shot assistant |
| `POST /sessions/{id}/interview` | Run the human-in-the-loop interviewer |

## Simple one-shot assistant

```bash
SID=$(curl -s -X POST localhost:8000/sessions | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

# Optional: subscribe to token/completed/error events in another terminal.
curl -N localhost:8000/sessions/$SID/events

# Normal JSON request/response.
curl -s -X POST localhost:8000/sessions/$SID/answer \
  -H 'content-type: application/json' \
  -d '{"question": "What is a vector database in one sentence?"}'
```

Browser clients can use the standard `EventSource` API:

```js
const events = new EventSource(`/sessions/${sessionId}/events`)
events.addEventListener("token", (event) => console.log(event.data))
events.addEventListener("completed", (event) => console.log(JSON.parse(event.data)))
events.addEventListener("input_required", (event) => console.log(JSON.parse(event.data)))
events.addEventListener("error", (event) => console.error(JSON.parse(event.data)))
```

## Human-in-the-loop (pause / resume)

The interview agent pauses to ask clarifying questions. The first POST sends the
initial request; if the agent needs more, the normal HTTP response is
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

## What it shows

- Writing FastAPI endpoints as ordinary HTTP request handlers
- Running Sefia workflows with a CLI-like procedural session block
- Publishing `token`, `completed`, `input_required`, and `error` events through a separate SSE channel
- Keeping SSE wiring out of the application workflow body

## Note on sessions

Each session persists to disk under the example's `.local/` directory. Send
requests for a given session one at a time; concurrent turns against the *same*
session are not supported. Different sessions are independent and safe to run
concurrently: the shared human-input store binds the active session per task
(via a `ContextVar`), so overlapping requests do not see each other's state.

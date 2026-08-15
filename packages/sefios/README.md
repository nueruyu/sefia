# sefios

Opinionated Stack for building applications with the
[Sefia](https://pypi.org/project/sefia/) framework.

Where `sefia` is the core framework (decorators, session, tool system),
`sefios` layers the pieces an application actually ships with: session
scoping and persistence, state containers, built-in tools, and integration
wiring for CLI and HTTP front ends.

## Install

```bash
pip install 'sefios[litellm]'
```

Extras:

| Extra       | Pulls in                            | Purpose                          |
| ----------- | ----------------------------------- | -------------------------------- |
| `litellm`   | [`sefia-litellm`](https://pypi.org/project/sefia-litellm/) | LLM providers via LiteLLM        |
| `cli`       | [`sefia-typer`](https://pypi.org/project/sefia-typer/)     | Typer (CLI) integration          |
| `fastapi`   | [`sefia-fastapi`](https://pypi.org/project/sefia-fastapi/) | FastAPI (HTTP) integration       |
| `web`       | `ddgs`                              | Built-in web search tool         |
| `sqlite`    | `glyff-sqlite`                      | Restart-safe local persistence   |
| `file-store` | `glyff-file-store`                 | Inspectable JSON debug storage   |
| `all`       | all of the above                    |                                  |

## Persistence

The default `MemoryPersistenceProvider` keeps execution, session state, and the
session registry process-local. Install the `sqlite` extra and explicitly select
`SQLitePersistenceProvider` for restart-safe local persistence.
`FilePersistenceProvider` is an optional, debug-oriented JSON representation and
requires the `file-store` extra.

## Documentation

See the [repository](https://github.com/nueruyu/sefia) for the full README,
tutorial, and architecture docs.

## Status

Early development. APIs may change before v1.0.

## License

MIT

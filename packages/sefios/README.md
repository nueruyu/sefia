# sefios

Official Stack for building applications with the
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
| `all`       | all of the above                    |                                  |

## Documentation

See the [repository](https://github.com/nueruyu/sefia) for the full README,
tutorial, and architecture docs.

## Status

Early development. APIs may change before v1.0.

## License

MIT

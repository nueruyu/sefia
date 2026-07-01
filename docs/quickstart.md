# Quickstart

```bash
pip install 'sefios[litellm]'
export OPENAI_API_KEY=sk-...
```

An `@infer` function is an abstract method the LLM implements: the signature is the
contract, the docstring is the instruction, the body is `...`. An agent holds its tools
as fields (their public methods are the tools) and runs inside a session.

```python
import asyncio
from pathlib import Path

from pydantic import BaseModel
from sefia import infer
from sefios import SessionScope
from sefios.tools import WebSearchTool


class Report(BaseModel):
    topic: str
    summary: str
    sources: list[str]


class Researcher:
    def __init__(self, web: WebSearchTool):
        self._web = web                 # held dependency; its public methods are tools

    @infer
    async def run(self, topic: str) -> Report:
        """Research the topic with web search and produce a structured report."""
        ...


scope = SessionScope(session_dir=Path(".sessions"), model="gpt-4o")


async def main() -> None:
    agent = Researcher(web=WebSearchTool())
    async with scope.session(session_id="demo"):
        print((await agent.run("durable execution for LLM agents")).summary)


asyncio.run(main())
```

The body never runs: sefia sends the signature, docstring, and arguments to the model
and validates the reply into `Report`. `SessionScope` wires the client, the glyff
session, and a file store under `.sessions/`.

Next, the [tutorial](./tutorial.md) builds this into a human-in-the-loop agent that
pauses for approval and resumes over HTTP after a restart.

> Release-target API (pre-1.0); some surfaces still differ — see [DESIGN.md](../DESIGN.md).

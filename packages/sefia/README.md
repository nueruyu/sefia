# sefia

**S**tateless **E**ngraved **F**unction **I**nference **A**bstraction

sefia turns typed Python functions into durable, replayable LLM-backed calls.
An `@infer` function is an abstract method whose implementer is an LLM: the
signature is the input contract, the return type is the validated output
contract, the docstring is the instruction, and the body is `...`.

```python
import glyff
from pydantic import BaseModel
from sefia import Domain


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


workflow = Domain(glyff.Domain("com.example.summaries", version="1"))

@workflow.infer(name="summarize")
async def summarize(article: str) -> Summary:
    """Summarize the article for a technical audience; note key uncertainty."""
    ...
```

Because model and tool steps replay on re-invocation, a call can pause, resume
after a restart, and drive human-in-the-loop flows over ordinary
request/response handlers — with no workflow engine or graph DSL.

## Install

```bash
pip install sefia
```

`sefia` is the core framework: decorators, session, profiles, and the tool
system. It is model-provider-agnostic; to talk to an actual LLM provider, add
an adapter such as
[`sefia-litellm`](https://pypi.org/project/sefia-litellm/), or install the
batteries-included stack [`sefios`](https://pypi.org/project/sefios/).

## Documentation

See the [repository](https://github.com/nueruyu/sefia) for the full README,
tutorial, and architecture docs.

## Status

Early development. APIs may change before v1.0.

## License

MIT

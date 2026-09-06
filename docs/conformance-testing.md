# Testing extension implementations

Sefia publishes reusable pytest contracts for extension authors. Install the
testing extra, subclass the contract that matches the interface, and override
its abstract factory method. Inherited tests call that method directly; they do
not depend on fixture names in the consuming project.

```bash
pip install "sefia[testing]"
```

For example, a history backend supplies one fresh store per test:

```python
from typing_extensions import override

from sefia import HistoryStorage
from sefia.testing import HistoryStorageContract


class TestPostgresHistoryStorage(HistoryStorageContract):
    @override
    def make_history_storage(self) -> HistoryStorage:
        return PostgresHistoryStorage(...)
```

The core package exports these contracts:

| Contract | Required override | Return type |
| --- | --- | --- |
| `LLMClientContract` | `make_llm_client_case()` | `LLMClientCase` |
| `StreamingLLMClientContract` | `make_streaming_llm_client_case()` | `StreamingLLMClientCase` |
| `HistoryStorageContract` | `make_history_storage()` | `HistoryStorage` |
| `DecisionTransportContract` | `make_decision_transport_case()` | `DecisionTransportCase` |
| `ToolCollectorContract` | `make_tool_collector_case()` | `ToolCollectorCase` |

Apply `StreamingLLMClientContract` only to clients supporting streaming.
`DecisionTransportCase` supplies the request and completion together with the
client's text chunks, reasoning chunks, structured events, and expected observer
output events. Keep the script consistent with the completion's protocol; for
example, a native tool call streams `tool_calls` paths. Apply both result and
tool-call cases when the transport supports them.

`sefios[testing]` adds persistence contracts:

| Contract | Required override | Return type |
| --- | --- | --- |
| `SessionStorageContract` | `make_session_storage()` | `SessionStorage` |
| `SessionRegistryContract` | `make_session_registry()` | `SessionRegistry` |
| `ActiveSessionStoreContract` | `make_active_session_store()` | `ActiveSessionStore` |
| `PersistenceProviderContract` | `make_persistence_provider()` | `PersistenceProvider` |

Each test needs isolated resources. Within one persistence test, repeated factory
calls must reopen the same logical store. An in-memory implementation can return
the same object; a durable implementation should return a new handle to the same
backing resource.

Subclasses can use local autouse fixtures for setup and cleanup, then expose the
prepared resource through the override:

```python
from pathlib import Path

import pytest
from glyff_pydantic import PydanticSerializer
from typing_extensions import override

from sefios.storage import FileSessionStorage, SessionStorage
from sefios.testing import SessionStorageContract


class TestFileStorage(SessionStorageContract):
    _directory: Path

    @pytest.fixture(autouse=True)
    def prepare_directory(self, tmp_path: Path) -> None:
        self._directory = tmp_path

    @override
    def make_session_storage(self) -> SessionStorage:
        return FileSessionStorage(self._directory, PydanticSerializer())
```

Async contracts require pytest-asyncio; configure `asyncio_mode = "auto"` in the
consuming project's pytest configuration. Setup fixtures that acquire resources
can use `yield` to release them after the inherited test finishes.

These suites cover the behavior visible through the Sefia interfaces. A custom
`PersistenceProvider` that supplies its own glyff backend should also apply the
appropriate contracts from `glyff.testing` to that backend.

Consumer tests can use `make_function_info`, `make_decision_request`,
`make_step_context`, and `make_tool_call_request` from `sefia.testing`. These
factories provide isolated defaults while keeping behavior-specific values explicit,
so unrelated tests do not depend on every constructor field.

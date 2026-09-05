# Testing extension implementations

Sefia publishes reusable pytest contracts for extension authors. Install the
testing extra, subclass the contract that matches the interface, and provide the
fixture named by that contract.

```bash
pip install "sefia[testing]"
```

For example, a history backend supplies one fresh store per test:

```python
import pytest

from sefia import HistoryStorage
from sefia.testing import HistoryStorageContract


class TestPostgresHistoryStorage(HistoryStorageContract):
    @pytest.fixture
    def history_storage(self) -> HistoryStorage:
        return PostgresHistoryStorage(...)
```

The core package exports these contracts:

- `LLMClientContract`, using an `llm_client_case` fixture that returns
  `LLMClientCase`.
- `StreamingLLMClientContract`, using a `streaming_llm_client_case` fixture that
  returns `StreamingLLMClientCase`; apply it only to clients supporting streaming.
- `HistoryStorageContract`, using the `history_storage` fixture.
- `DecisionTransportContract`, using a `decision_transport_case` fixture that
  returns `DecisionTransportCase`.
- `ToolCollectorContract`, using a `tool_collector_case` fixture that returns
  `ToolCollectorCase`.

`sefios[testing]` adds persistence contracts:

- `SessionStorageContract`, using `session_storage_factory`.
- `SessionRegistryContract`, using `session_registry_factory`.
- `ActiveSessionStoreContract`, using `active_session_store_factory`.
- `PersistenceProviderContract`, using `persistence_provider`.

Factory fixtures must reopen the same logical store. An in-memory implementation
can return the same object; a durable implementation should return a new handle to
the same backing resource. Async contracts require pytest-asyncio; configure
`asyncio_mode = "auto"` in the consuming project's pytest configuration.

These suites cover the behavior visible through the Sefia interfaces. A custom
`PersistenceProvider` that supplies its own glyff backend should also apply the
appropriate contracts from `glyff.testing` to that backend.

Consumer tests can use `make_function_info`, `make_decision_request`,
`make_step_context`, and `make_tool_call_request` from `sefia.testing`. These
factories provide isolated defaults while keeping behavior-specific values explicit,
so unrelated tests do not depend on every constructor field.

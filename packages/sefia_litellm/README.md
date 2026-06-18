# sefia_litellm

A `sefia` `LLMClient` implementation that connects to various LLM providers via
[LiteLLM](https://github.com/BerriAI/litellm).

```python
import sefia_litellm

client = sefia_litellm.LiteLLMClient(model="gpt-4o")
```

## Suppressing LiteLLM logging

LiteLLM emits verbose logging by default (the `Provider List: ...` banner and
debug info printed alongside exceptions). `LiteLLMClient` **suppresses these by
default**.

There are two ways to control this:

- **Constructor argument** `suppress_logs` (takes precedence)

  ```python
  # Suppress logs (default)
  client = sefia_litellm.LiteLLMClient(model="gpt-4o")
  client = sefia_litellm.LiteLLMClient(model="gpt-4o", suppress_logs=True)

  # Let LiteLLM log as usual
  client = sefia_litellm.LiteLLMClient(model="gpt-4o", suppress_logs=False)
  ```

- **Environment variable** `SEFIA_LITELLM_SUPPRESS_LOGS` (used as the default when
  `suppress_logs` is `None`)

  ```bash
  # Disable suppression (let logs through). 0/false/no/off disable it.
  export SEFIA_LITELLM_SUPPRESS_LOGS=false
  ```

  An explicit `suppress_logs` argument overrides the environment variable. When
  unset, suppression is on.

When suppression is on, `litellm.suppress_debug_info` is set to `True` and the
standard-library `"LiteLLM"` logger is raised to `WARNING` (silencing INFO/DEBUG).

## On slow imports

Importing LiteLLM is heavy and can take around a second
(see [BerriAI/litellm#7605](https://github.com/BerriAI/litellm/issues/7605)).

This package mitigates that as follows:

1. **Lazy import** — LiteLLM is imported only inside the methods that actually
   send a request. Importing `sefia_litellm` alone costs nothing. After the first
   request the module is cached in `sys.modules`, so subsequent imports are
   effectively free. This is the primary mitigation.

2. **Local model cost map** — `LITELLM_LOCAL_MODEL_COST_MAP=True` is set before
   LiteLLM is imported so that it uses its bundled cost map JSON instead of
   fetching it over the network. This speeds up the import and keeps it working
   offline.

   If you need up-to-date pricing for the newest models and the bundled map is
   stale, you can restore the original behavior:

   ```bash
   export LITELLM_LOCAL_MODEL_COST_MAP=False
   ```

3. **(Optional) Warm up at startup** — if you also want to hide the first
   request's latency, import LiteLLM in the background during application startup.
   A daemon thread works regardless of whether an asyncio event loop is running
   yet:

   ```python
   import threading

   threading.Thread(target=__import__, args=("litellm",), daemon=True).start()
   ```

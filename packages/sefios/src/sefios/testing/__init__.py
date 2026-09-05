"""Public conformance contracts for sefios persistence extensions.

Subclass the applicable contract and provide its factory fixture. Install the
``testing`` extra to use these pytest-based contracts.
"""

from ._active_session_store_contract import (
    ActiveSessionStoreContract,
    ActiveSessionStoreFactory,
)
from ._persistence_contract import PersistenceProviderContract
from ._session_registry_contract import SessionRegistryContract, SessionRegistryFactory
from ._session_storage_contract import SessionStorageContract, SessionStorageFactory

__all__ = [
    "ActiveSessionStoreContract",
    "ActiveSessionStoreFactory",
    "PersistenceProviderContract",
    "SessionRegistryContract",
    "SessionRegistryFactory",
    "SessionStorageContract",
    "SessionStorageFactory",
]

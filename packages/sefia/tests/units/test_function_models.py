import gc
import weakref

from sefia.pydantic._function_models import PydanticFunctionModelFactory, cache_key


def _sample_func(a: int, b: str = "x") -> bool:
    """Sample function."""
    return True


class _UnhashableCallable:
    def __call__(self, value: int) -> str:
        return str(value)

    def __eq__(self, other: object) -> bool:
        return self is other


def test_cache_key_uses_identity_for_unhashable_callables():
    func = _UnhashableCallable()

    key1 = cache_key(func)
    key2 = cache_key(func)
    other_key = cache_key(_UnhashableCallable())

    assert key1 == key2
    assert hash(key1) == hash(key2)
    assert key1 != other_key


def test_cache_key_keeps_unhashable_callables_alive():
    func = _UnhashableCallable()
    ref = weakref.ref(func)
    key = cache_key(func)

    del func
    gc.collect()

    assert key is not None
    assert ref() is not None


def test_function_model_factory_reuses_params_model():
    factory = PydanticFunctionModelFactory()

    model1 = factory.params_model(
        _sample_func,
        name="_sample_func",
        extra="forbid",
    )
    model2 = factory.params_model(
        _sample_func,
        name="_sample_func",
        extra="forbid",
    )

    assert model1 is model2

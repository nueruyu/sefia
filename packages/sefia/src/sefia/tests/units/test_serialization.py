from dataclasses import dataclass

from pydantic import BaseModel

from sefia.serialization import SefiaArgsHasher, SefiaSerializer


class _Model(BaseModel):
    a: int
    b: str


@dataclass(frozen=True)
class _D:
    value: int


def _fn(x: int, y: str = "a") -> str:
    return f"{x}-{y}"


class TestSefiaSerializer:
    def test_roundtrip_pydantic_model(self):
        serializer = SefiaSerializer()
        value = _Model(a=1, b="x")

        data = serializer.serialize(value, _Model)
        restored = serializer.deserialize(data, _Model)

        assert isinstance(restored, _Model)
        assert restored.a == 1
        assert restored.b == "x"

    def test_roundtrip_dataclass(self):
        serializer = SefiaSerializer()
        value = _D(value=7)

        data = serializer.serialize(value, _D)
        restored = serializer.deserialize(data, _D)

        assert isinstance(restored, _D)
        assert restored.value == 7


class TestSefiaArgsHasher:
    def test_hash_is_stable_for_same_args(self):
        hasher = SefiaArgsHasher()
        sig = __import__("inspect").signature(_fn)

        h1 = hasher.hash_args(_fn, sig, (1,), {"y": "b"})
        h2 = hasher.hash_args(_fn, sig, (1,), {"y": "b"})

        assert h1 == h2

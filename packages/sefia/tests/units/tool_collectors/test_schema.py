from sefia.pydantic import PydanticModelBackend


class WebToolkit:
    """A two-method toolkit; discovery only reads its names and docstrings."""

    async def search(self, query: str) -> str:
        """Search the web for a query."""
        raise NotImplementedError

    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Schema generation (unchanged surface)
# --------------------------------------------------------------------------- #


def example_func(a: int, b: str = "default") -> bool:
    """An example function."""
    return True


def test_create_tool_schema_from_function():
    definition = PydanticModelBackend().definition(example_func, name="example_func")

    assert definition.name == "example_func"
    assert definition.description == "An example function."

    params = definition.parameters
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "string"
    assert params["properties"]["b"]["default"] == "default"
    assert "a" in params["required"]
    assert "b" not in params.get("required", [])


def test_schema_builder_sanitizes_complex_names():
    backend = PydanticModelBackend()

    class Outer:
        class Inner:
            def my_method(self):
                pass

    name = backend.tool_name(Outer.Inner.my_method)
    assert name.endswith("Outer_Inner_my_method")
    assert "." not in name
    assert "<" not in name


def test_schema_builder_caches_results():
    backend = PydanticModelBackend()
    definition1 = backend.definition(example_func, name="example_func")
    definition2 = backend.definition(example_func, name="example_func")
    assert definition1 is definition2

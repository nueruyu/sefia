from typing import Annotated, Optional, Protocol, TypeVar

from typing_extensions import TypeAliasType

from sefia._introspection import (
    declared_fields,
    declared_methods,
    is_protocol,
    unwrap_annotation,
)
from sefia._tool_system import Tools, bears_tools, role_interface


class WebToolkit:
    async def search(self, q: str) -> list[str]:
        """Search the web."""
        return []

    def _internal(self) -> None: ...


# --------------------------------------------------------------------------- #
# unwrap_role / bears_tools / role_interface
# --------------------------------------------------------------------------- #


def test_plain_types_bear_no_role():
    assert not bears_tools(WebToolkit)
    assert not bears_tools(str)
    assert role_interface(WebToolkit) is WebToolkit


def test_subscripted_alias_bears_the_role_and_yields_the_interface():
    assert bears_tools(Tools[WebToolkit])
    assert role_interface(Tools[WebToolkit]) is WebToolkit


def test_optional_around_the_alias_is_unwrapped():
    assert bears_tools(Optional[Tools[WebToolkit]])
    assert role_interface(Optional[Tools[WebToolkit]]) is WebToolkit
    # ... and Optional inside user aliases of the alias, via __value__.
    assert bears_tools(Tools[WebToolkit] | None)


def test_stacked_annotated_metadata_over_the_alias_still_resolves():
    hint = Annotated[Tools[WebToolkit], "other metadata"]
    assert bears_tools(hint)
    assert role_interface(hint) is WebToolkit


def test_non_role_annotated_metadata_bears_no_role():
    assert not bears_tools(Annotated[WebToolkit, "metadata"])
    assert role_interface(Annotated[WebToolkit, "metadata"]) is WebToolkit


MyKit = TypeAliasType("MyKit", Tools[WebToolkit])  # a user alias over the role alias

_T = TypeVar("_T")
GenericKit = TypeAliasType("GenericKit", Tools[_T], type_params=(_T,))


def test_a_user_alias_wrapping_the_role_alias_resolves():
    assert bears_tools(MyKit)
    assert role_interface(MyKit) is WebToolkit


def test_a_generic_user_alias_substitutes_its_type_argument():
    assert bears_tools(GenericKit[WebToolkit])
    assert role_interface(GenericKit[WebToolkit]) is WebToolkit


def test_an_unsubscripted_generic_user_alias_still_bears_the_role():
    # No argument to substitute: the body's TypeVar remains — the marker is
    # found, but there is no class interface to expose (fail-closed).
    assert bears_tools(GenericKit)
    assert not isinstance(role_interface(GenericKit), type)


def test_a_bare_unsubscripted_alias_yields_no_class_interface():
    # ``_web: Tools`` (no argument) unwraps to the alias body's TypeVar —
    # not a class, so the collector's isclass guard rejects it (fail-closed).
    metadata, interface = unwrap_annotation(Tools)
    assert metadata  # the role marker is present...
    assert not isinstance(interface, type)  # ...but there is nothing to expose


def test_an_ambiguous_union_stops_resolution():
    _, interface = unwrap_annotation(Tools[WebToolkit] | str)
    assert not isinstance(interface, type)


# --------------------------------------------------------------------------- #
# exposed_methods
# --------------------------------------------------------------------------- #


class Concrete:
    async def visible(self) -> str:
        return "ok"

    def _hidden(self) -> None: ...

    @staticmethod
    async def static_tool(q: str) -> str:
        return q

    @classmethod
    async def class_tool(cls, q: str) -> str:
        return q

    @property
    def prop(self) -> str:
        raise AssertionError("property getters must not run during discovery")


class Proto(Protocol):
    async def visible(self) -> str: ...

    async def _also_visible(self) -> str: ...


def test_a_concrete_class_declares_only_public_methods():
    names = set(declared_methods(Concrete))
    assert names == {"visible", "static_tool", "class_tool"}


def test_a_protocol_declares_its_private_members_too():
    assert set(declared_methods(Proto)) == {"visible", "_also_visible"}


def test_inherited_methods_are_included_and_overrides_win():
    class Base:
        async def base_method(self) -> str:
            return "base"

        async def shared(self) -> str:
            return "base"

    class Child(Base):
        async def shared(self) -> str:
            return "child"

    methods = declared_methods(Child)
    assert set(methods) == {"base_method", "shared"}
    assert methods["shared"] is Child.__dict__["shared"]


def test_is_protocol():
    assert is_protocol(Proto)
    assert not is_protocol(Concrete)


class FieldBase:
    _base_field: WebToolkit


class FieldChild(FieldBase):
    _child_field: Tools[WebToolkit]
    _unresolvable: "NoSuchName"  # noqa: F821  # pyright: ignore[reportUndefinedVariable]

    @property
    def _narrowed(self) -> WebToolkit: ...  # a read-only property declaration


def test_declared_fields_collects_annotations_across_the_mro():
    fields = declared_fields(FieldChild)
    assert fields["_base_field"] is WebToolkit
    assert bears_tools(fields["_child_field"])


def test_an_unresolvable_annotation_is_skipped_not_widened():
    assert "_unresolvable" not in declared_fields(FieldChild)


def test_a_readonly_property_declares_a_field_via_its_return_type():
    assert declared_fields(FieldChild)["_narrowed"] is WebToolkit


def test_string_annotations_resolve_against_the_defining_module():
    class Stringly:
        _web: "WebToolkit"

    assert declared_fields(Stringly)["_web"] is WebToolkit

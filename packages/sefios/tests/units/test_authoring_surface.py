"""`sefios` re-exports the core authoring surface as the same objects."""

import sefia
import sefios


def test_reexports_are_the_source_objects():
    assert sefios.concurrent is sefia.concurrent
    assert sefios.preview is sefia.preview
    assert sefios.policy is sefia.policy
    assert sefios.profile is sefia.profile
    assert sefios.Policy is sefia.Policy
    assert sefios.Profile is sefia.Profile
    assert sefios.Tools is sefia.Tools


def test_authoring_surface_is_public():
    assert "infer" not in sefios.__all__
    for name in (
        "concurrent",
        "preview",
        "policy",
        "profile",
        "domain",
        "Policy",
        "Profile",
        "Tools",
    ):
        assert name in sefios.__all__


def test_exceptions_are_not_reexported_from_package_root():
    assert "InputRequired" not in sefios.__all__
    assert not hasattr(sefios, "InputRequired")
